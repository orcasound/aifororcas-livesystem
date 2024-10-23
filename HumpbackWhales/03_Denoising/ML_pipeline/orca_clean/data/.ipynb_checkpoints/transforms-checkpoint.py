"""
Module: transforms.py
Authors: Christian Bergler
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import io
import os
import sys
import math
import resampy
import numpy as np
import scipy.fftpack
import soundfile as sf

import torch
import torch.nn.functional as F

from typing import List
from multiprocessing import Lock
from utils.FileIO import AsyncFileReader, AsyncFileWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

""" Load audio file  """
def load_audio_file(file_name, sr=None, mono=True):
    y, sr_orig = sf.read(file_name, always_2d=True, dtype="float32")
    if mono and y.ndim == 2 and y.shape[1] > 1:
        y = np.mean(y, axis=1, keepdims=True)
    if sr is not None and sr != sr_orig:
        y = resampy.resample(y, sr_orig, sr, axis=0, filter="kaiser_best")
    return torch.from_numpy(y).float().t()

"""Composes several transforms to one."""
class Compose(object):

    def __init__(self, *transforms):
        if len(transforms) == 1 and isinstance(transforms[0], list):
            self.transforms = transforms[0]
        else:
            self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

"""Pre-Emphasize in order to raise higher frequencies and lower low frequencies."""
class PreEmphasize(object):
    def __init__(self, factor=0.97):
        self.factor = factor

    def __call__(self, y):
        if y.dim() != 2:
            raise ValueError(
                "PreEmphasize expects a 2 dimensional signal of size (c, n), "
                "but got size: {}.".format(y.size())
            )
        return torch.cat(
            (y[:, 0].unsqueeze(dim=-1), y[:, 1:] - self.factor * y[:, :-1]), dim=-1
        )

"""Converts a given audio to a spectrogram."""
class Spectrogram(object):


    def __init__(self, n_fft, hop_length, center=True):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.window = torch.hann_window(self.n_fft)

    def __call__(self, y):
        if y.dim() != 2:
            raise ValueError(
                "Spectrogram expects a 2 dimensional signal of size (c, n), "
                "but got size: {}.".format(y.size())
            )
        S = torch.stft(
            input=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            onesided=True,
            return_complex=False
        ).transpose(1, 2)
        Sp = S/(self.window.pow(2).sum().sqrt())
        Sp = Sp.pow(2).sum(-1)
        return Sp, S


"""Converts a given audio to a spectrogram, cache and store the spectrograms."""
class CachedSpectrogram(object):
    version = 4

    def __init__(
        self, cache_dir, spec_transform, file_reader=None, file_writer=None, **meta
    ):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        if file_reader is not None:
            self.reader = file_reader
        else:
            self.reader = AsyncFileReader(n_readers=1)
        self.transform = spec_transform
        self.meta = meta
        if file_writer is not None:
            self.writer = file_writer
        else:
            self.writer = AsyncFileWriter(write_fn=self._write_fn, n_writers=1)

    def get_cached_name(self, file_name):
        cached_spec_n = os.path.splitext(os.path.basename(file_name))[0] + ".spec"
        dir_structure = os.path.dirname(file_name).replace(r"/", "_") + "_"
        cached_spec_n = dir_structure + cached_spec_n
        if not os.path.isabs(cached_spec_n):
            cached_spec_n = os.path.join(self.cache_dir, cached_spec_n)
        return cached_spec_n

    def __call__(self, fn):
        cached_spec_n = self.get_cached_name(fn)
        if not os.path.isfile(cached_spec_n):
            return self._compute_and_cache(fn)
        try:
            data = self.reader(cached_spec_n)
            spec_dict = torch.load(io.BytesIO(data), map_location="cpu")
        except (EOFError, RuntimeError):
            return self._compute_and_cache(fn)
        if not (
            "v" in spec_dict
            and spec_dict["v"] == self.version
            and "data" in spec_dict
            and spec_dict["data"].dim() == 3
        ):
            return self._compute_and_cache(fn)
        for key, value in self.meta.items():
            if not (key in spec_dict and spec_dict[key] == value):
                return self._compute_and_cache(fn)
        return spec_dict["data"], None  # 2 values needed to unpack (#FIXME)

    def _compute_and_cache(self, fn):
        try:
            audio_data = self.reader(fn)
            spec, spec_cmplx = self.transform(io.BytesIO(audio_data))
        except Exception:
            spec, spec_cmplx = self.transform(fn)
        self.writer(self.get_cached_name(fn), spec)
        return spec, spec_cmplx

    def _write_fn(self, fn, data):
        spec_dict = {"v": self.version, "data": data}
        for key, value in self.meta.items():
            spec_dict[key] = value
        torch.save(spec_dict, fn)

"""Normalize db scale to 0..1"""
class Normalize(object):

    def __init__(self, min_level_db=-100, ref_level_db=20):
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db

    def __call__(self, spec):
        return torch.clamp(
            (spec - self.ref_level_db - self.min_level_db) / -self.min_level_db, 0, 1
        )

"""Normalize min/max scale to 0..1"""
class MinMaxNormalize(object):

    def __call__(self, spectrogram):
        spectrogram -= spectrogram.min()
        if spectrogram.max().item() == 0.0:
            return spectrogram
        spectrogram /= spectrogram.max()
        return spectrogram

"""Turns a spectrogram from the power/amplitude scale to the decibel scale.

Code from https://github.com/pytorch/audio/blob/5787787edc/torchaudio/transforms.py

BSD 2-Clause License

Copyright (c) 2017 Facebook Inc. (Soumith Chintala), 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Access Data: 06.02.2021, Last Access Date: 21.12.2021
Changes: Modified by Christian Bergler (06.02.2021)
"""
class Amp2Db(object):

    def __init__(self, min_level_db=None, stype="power"):
        self.stype = stype
        self.multiplier = 10. if stype == "power" else 20.
        if min_level_db is None:
            self.min_level = None
        else:
            min_level_db = -min_level_db if min_level_db > 0 else min_level_db
            self.min_level = torch.tensor(
                np.exp(min_level_db / self.multiplier * np.log(10))
            )

    def __call__(self, spec):
        if self.min_level is not None:
            spec_ = torch.max(spec, self.min_level)
        else:
            spec_ = spec
        spec_db = self.multiplier * torch.log10(spec_)
        return spec_db

"""Scaling spectrogram dimension (time/frequency) by a given factor."""
def _scale(spectrogram: torch.Tensor, shift_factor: float, dim: int):
    in_dim = spectrogram.dim()
    if in_dim < 3:
        raise ValueError(
            "Expected spectrogram with size (c t f) or (n c t f)"
            ", but got {}".format(spectrogram.size())
        )
    if in_dim == 3:
        spectrogram.unsqueeze_(dim=0)
    size = list(spectrogram.shape)[2:]
    dim -= 1
    size[dim] = int(round(size[dim] * shift_factor))
    spectrogram = F.interpolate(spectrogram, size=size, mode="nearest")
    if in_dim == 3:
        spectrogram.squeeze_(dim=0)
    return spectrogram


"""Randomly shifts the pitch of a spectrogram by a factor of 2**Uniform(log2(from), log2(to))."""
class RandomPitchSift(object):

    def __init__(self, from_=0.5, to_=1.5):
        self.from_ = math.log2(from_)
        self.to_ = math.log2(to_)

    def __call__(self, spectrogram: torch.Tensor):
        factor = 2 ** torch.empty((1,)).uniform_(self.from_, self.to_).item()
        median = spectrogram.median()
        size = list(spectrogram.shape)
        scaled = _scale(spectrogram, factor, dim=2)
        if factor > 1:
            out = scaled[:, :, : size[2]]
        else:
            out = torch.full(size, fill_value=median, dtype=spectrogram.dtype)
            new_f_bins = int(round(size[2] * factor))
            out[:, :, 0:new_f_bins] = scaled
        return out

"""Randomly stretches the time of a spectrogram by a factor of 2**Uniform(log2(from), log2(to))."""
class RandomTimeStretch(object):

    def __init__(self, from_=0.5, to_=2):
        self.from_ = math.log2(from_)
        self.to_ = math.log2(to_)

    def __call__(self, spectrogram: torch.Tensor):
        factor = 2 ** torch.empty((1,)).uniform_(self.from_, self.to_).item()
        return _scale(spectrogram, factor, dim=1)

"""Randomly scaling (uniform distributed) the amplitude based on a given input spectrogram (intensity augmenation)."""
class RandomAmplitude(object):
    def __init__(self, increase_db=3, decrease_db=None):
        self.inc_db = increase_db
        if decrease_db is None:
            decrease_db = -increase_db
        elif decrease_db > 0:
            decrease_db *= -1
        self.dec_db = decrease_db

    def __call__(self, spec):
        db_change = torch.randint(
            self.dec_db, self.inc_db, size=(1,), dtype=torch.float
        )
        return spec.mul(10 ** (db_change / 10))

""" 
Randomly adds a given noise file to the given spectrogram by considering a randomly selected
(uniform distributed) SNR of min = -3 dB and max = 12 dB. The noise file could also be intensity, pitch, and/or time
augmented. If a noise file is longer/shorter than the given spectrogram it will be subsampled/self-concatenated. 
The spectrogram is expected to be a power spectrogram, which is **not** logarithmically compressed.
"""
class RandomAddNoise(object):

    def __init__(
        self,
        noise_files: List[str],
        spectrogram_transform,
        transform,
        min_length=0,
        min_snr=12,
        max_snr=-3,
        return_original=False,
    ):
        if not noise_files:
            raise ValueError("No noise files found")
        self.noise_files = noise_files
        self.t_spectrogram = spectrogram_transform
        self.noise_file_locks = {file: Lock() for file in noise_files}
        self.transform = transform
        self.min_length = min_length
        self.t_pad = PaddedSubsequenceSampler(sequence_length=min_length, dim=1)
        self.min_snr = min_snr if min_snr > max_snr else max_snr
        self.max_snr = max_snr if min_snr > max_snr else min_snr
        self.return_original = return_original

    def __call__(self, spectrogram):
        if len(self.noise_files) == 1:
            idx = 0
        else:
            idx = torch.randint(
                0, len(self.noise_files) - 1, size=(1,), dtype=torch.long
            ).item()
        noise_file = self.noise_files[idx]

        try:
            if not self.noise_file_locks[noise_file].acquire(timeout=10):
                print("Warning: Could not acquire lock for {}".format(noise_file))
                return spectrogram, None # 2 values needed to unpack (#FIXME)
            noise_spec, _ = self.t_spectrogram(noise_file)
        except Exception:
            import traceback

            print(traceback.format_exc())
            return spectrogram, None # 2 values needed to unpack (#FIXME)
        finally:
            self.noise_file_locks[noise_file].release()

        noise_spec = self.t_pad._maybe_sample_subsequence(
            noise_spec, spectrogram.size(1) * 2
        )
        noise_spec = self.transform(noise_spec)

        if self.min_length > 0:
            spectrogram = self.t_pad._maybe_pad(spectrogram)

        if spectrogram.size(1) > noise_spec.size(1):
            n_repeat = int(math.ceil(spectrogram.size(1) / noise_spec.size(1)))
            noise_spec = noise_spec.repeat(1, n_repeat, 1)
        if spectrogram.size(1) < noise_spec.size(1):
            high = noise_spec.size(1) - spectrogram.size(1)
            start = torch.randint(0, high, size=(1,), dtype=torch.long)
            end = start + spectrogram.size(1)
            noise_spec_part = noise_spec[:, start:end]
        else:
            noise_spec_part = noise_spec

        if self.max_snr == self.min_snr:
            snr = self.max_snr
        else:
            snr = torch.randint(self.max_snr, self.min_snr, size=(1,), dtype=torch.float)

        signal_power = spectrogram.sum()
        noise_power = noise_spec_part.sum()

        K = (signal_power / noise_power) * 10 ** (-snr / 10)
        spectrogram_aug = spectrogram + noise_spec_part * K

        if self.return_original:
            return spectrogram_aug, spectrogram
        return spectrogram_aug, None # 2 values needed to unpack (#FIXME)

"""Samples a subsequence along one axis and pads if necessary."""
class PaddedSubsequenceSampler(object):

    def __init__(self, sequence_length: int, dim: int = 0, random=True):
        assert isinstance(sequence_length, int)
        assert isinstance(dim, int)
        self.sequence_length = sequence_length
        self.dim = dim
        if random:
            self._sampler = lambda x: torch.randint(
                0, x, size=(1,), dtype=torch.long
            ).item()
        else:
            self._sampler = lambda x: x // 2

    def _maybe_sample_subsequence(self, spectrogram, sequence_length=None):
        if sequence_length is None:
            sequence_length = self.sequence_length
        sample_length = spectrogram.shape[self.dim]
        if sample_length > sequence_length:
            start = self._sampler(sample_length - sequence_length)
            end = start + sequence_length
            indices = torch.arange(start, end, dtype=torch.long)
            return torch.index_select(spectrogram, self.dim, indices)
        return spectrogram

    def _maybe_pad(self, spectrogram, sequence_length=None):
        if sequence_length is None:
            sequence_length = self.sequence_length
        sample_length = spectrogram.shape[self.dim]
        if sample_length < sequence_length:
            start = self._sampler(sequence_length - sample_length)
            end = start + sample_length

            shape = list(spectrogram.shape)
            shape[self.dim] = sequence_length
            padded_spectrogram = torch.zeros(shape, dtype=spectrogram.dtype)

            if self.dim == 0:
                padded_spectrogram[start:end] = spectrogram
            elif self.dim == 1:
                padded_spectrogram[:, start:end] = spectrogram
            elif self.dim == 2:
                padded_spectrogram[:, :, start:end] = spectrogram
            elif self.dim == 3:
                padded_spectrogram[:, :, :, start:end] = spectrogram
            return padded_spectrogram
        return spectrogram

    def __call__(self, spectrogram):
        spectrogram = self._maybe_pad(spectrogram)
        spectrogram = self._maybe_sample_subsequence(spectrogram)
        return spectrogram

"""Frequency compression of a given frequency range into a chosen number of frequency bins."""
class Interpolate(object):
    def __init__(self, n_freqs, sr=None, f_min=0, f_max=None):
        self.n_freqs = n_freqs
        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max

    def __call__(self, spec):
        n_fft = (spec.size(2) - 1) * 2

        if self.sr is not None and n_fft is not None:
            min_bin = int(max(0, math.floor(n_fft * self.f_min / self.sr)))
            max_bin = int(min(n_fft - 1, math.ceil(n_fft * self.f_max / self.sr)))
            spec = spec[:, :, min_bin:max_bin]

        spec.unsqueeze_(dim=0)
        spec = F.interpolate(spec, size=(spec.size(2), self.n_freqs), mode="nearest")
        return spec.squeeze(dim=0)

"""Frequency decompression of a given frequency range into a chosen number of frequency bins (important for reconstruction of the cmplx spectrogram)."""
class Decompress(object):
    def __init__(self, f_min=0, f_max=10000, n_fft=4096, sr=44100):
        self.sr = sr
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max

    def __call__(self, spectrogram):
        min_bin = int(max(0, math.floor(self.n_fft * self.f_min / self.sr)))
        max_bin = int(min(self.n_fft - 1, math.ceil(self.n_fft * self.f_max / self.sr)))

        spec = F.interpolate(spectrogram, size=(spectrogram.size(2), max_bin - min_bin), mode="nearest").squeeze(dim=0)
        lower_spec = torch.zeros([1, spectrogram.size(2), min_bin])
        upper_spec = torch.zeros([1, spectrogram.size(2), (self.n_fft // 2 + 1) - max_bin])

        final_spec = torch.cat((lower_spec, spec), 2)
        final_spec = torch.cat((final_spec, upper_spec), 2)

        return final_spec


"""Convert hertz to mel."""
def _hz2mel(f):
    return 2595 * np.log10(1 + f / 700)

"""Convert mel to hertz."""
def _mel2hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

"""
Create melbank.
Code from https://github.com/pytorch/audio/blob/5787787edc/torchaudio/transforms.py

BSD 2-Clause License

Copyright (c) 2017 Facebook Inc. (Soumith Chintala), 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Access Data: 06.02.2021, Last Access Date: 21.12.2021
Changes: Modified by Christian Bergler (06.02.2021)
"""
def _melbank(sr, n_fft, n_mels=128, f_min=0.0, f_max=None, inverse=False):
    m_min = 0. if f_min == 0 else _hz2mel(f_min)
    m_max = _hz2mel(f_max if f_max is not None else sr // 2)

    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel2hz(m_pts)

    bins = torch.floor(((n_fft - 1) * 2 + 1) * f_pts / sr).long()

    fb = torch.zeros(n_mels, n_fft)
    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1].item()
        f_m = bins[m].item()
        f_m_plus = bins[m + 1].item()

        if f_m_minus != f_m:
            fb[m - 1, f_m_minus:f_m] = (torch.arange(f_m_minus, f_m) - f_m_minus).float() / (
                f_m - f_m_minus
            )
        if f_m != f_m_plus:
            fb[m - 1, f_m:f_m_plus] = (f_m_plus - torch.arange(f_m, f_m_plus)).float() / (
                f_m_plus - f_m
            )

    if not inverse:
        return fb.t()
    else:
        return fb


"""
This turns a normal STFT into a MEL Frequency STFT, using a conversion matrix.  This uses triangular filter banks.
Code from https://github.com/pytorch/audio/blob/5787787edc/torchaudio/transforms.py

BSD 2-Clause License

Copyright (c) 2017 Facebook Inc. (Soumith Chintala), 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Access Data: 06.02.2021, Last Access Date: 21.12.2021
Changes: Modified by Christian Bergler (06.02.2021)
"""
class F2M(object):


    def __init__(
        self, sr: int = 16000, n_mels: int = 40, f_min: int = 0, f_max: int = None
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sr // 2
        if self.f_max > self.sr // 2:
            raise ValueError("f_max > sr // 2")

    def __call__(self, spec_f: torch.Tensor):
        n_fft = spec_f.size(2)

        fb = _melbank(self.sr, n_fft, self.n_mels, self.f_min, self.f_max)

        spec_m = torch.matmul(
            spec_f, fb
        )
        return spec_m


"""
Converts a normal STFT into a MEL Frequency STFT, using a conversion
matrix. This uses triangular filter banks.
"""
class M2F(object):

    def __init__(
        self, sr: int = 16000, n_fft: int = 1024, f_min: int = 0, f_max: int = None
    ):
        self.sr = sr
        self.n_fft = n_fft // 2 + 1
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sr // 2
        if self.f_max > self.sr // 2:
            raise ValueError("f_max > sr // 2")

    def __call__(self, spec_m: torch.Tensor):
        n_mels = spec_m.size(2)

        fb = _melbank(self.sr, self.n_fft, n_mels, self.f_min, self.f_max, inverse=True)

        spec_f = torch.matmul(
            spec_m, fb
        )
        return spec_f

"""
Converts MEL Frequency to MFCC.
"""
class M2MFCC(object):

    def __init__(self, n_mfcc : int = 32):
        self.n_mfcc = n_mfcc

    def __call__(self, spec_m):
        device = spec_m.device
        spec_m = 10 * torch.log10(spec_m)
        spec_m[spec_m == float('-inf')] = 0
        if isinstance(spec_m, torch.Tensor):
            spec_m = spec_m.cpu().numpy()
        mfcc = scipy.fftpack.dct(spec_m, axis=-1)
        mfcc = mfcc[:, :, 1:self.n_mfcc+1]
        return torch.from_numpy(mfcc).to(device)
