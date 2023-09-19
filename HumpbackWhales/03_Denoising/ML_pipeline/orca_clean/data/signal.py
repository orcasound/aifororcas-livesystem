"""
Module: signal.py
Authors: Christian Bergler
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import cv2
import math
import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

matplotlib.use('Agg')


from typing import Tuple

from scipy import ndimage as ndi
from scipy.signal import find_peaks

from skimage import filters
from skimage import exposure

from visualization import utils as U

from visualization.cm import apply_cm, viridis_cm

"""
Signal processing functions in order to detect spectral strong regions
as well as computing spectral masks
"""
class signal_proc(object):

    def __init__(self):
        pass

    """ Convert numpy array to pytorch tensor  """
    def np_array_to_tensor(self, audio_data):
        audio_data = torch.from_numpy(audio_data).float().t()
        return audio_data

    """ Calculate mean of specific frequency range without considering zero-elements  """
    def get_mean_from_to(self, power_spectrogram, min_freq_bin, max_freq_bin):
        sub_power_spec = power_spectrogram.clone()
        part_of_spec = sub_power_spec[:, min_freq_bin:max_freq_bin]
        non_zero_elements_in_spec = self.get_non_zero_elements_tensor(part_of_spec)
        sum_spec_values = part_of_spec.sum().float()
        if non_zero_elements_in_spec != 0:
            mean_part_of_spec = sum_spec_values / non_zero_elements_in_spec
        else:
            mean_part_of_spec = 0
        return mean_part_of_spec

    """ Calculate mean of spectrogram without considering zero-elements  """
    def get_mean_spectrogram(self, power_spectrogram):
        non_zero_elements_in_spec = self.get_non_zero_elements_tensor(power_spectrogram)
        sum_spec_values = power_spectrogram.sum().float()
        mean = (sum_spec_values / non_zero_elements_in_spec)
        return mean

    """ Calculate number of non-zero elements in tensor  """
    def get_non_zero_elements_tensor(self, input_tensor):
        non_zero_elements = 0
        for element in input_tensor:
            non_zero_elements += torch.nonzero(element).size()[0]
        return non_zero_elements

    """ Calculate a self potentiated version of a specific frequency region of the original spectrogram  """
    def insert_non_linearity(self, power_spectrogram, min_freq_bin, max_freq_bin, exponent):
        non_linear_power_spec = power_spectrogram.clone()
        n_freq_bins = non_linear_power_spec[:, min_freq_bin:max_freq_bin]
        n_freq_bins = torch.pow(n_freq_bins, exponent)
        non_linear_power_spec[:, min_freq_bin:max_freq_bin] = n_freq_bins
        return non_linear_power_spec

    """ Clear spectral values of a spectrogram between a certain frequency range by substracting a weighted valued or setting value to zero if below a certain weighted values """
    def clear_val_from_to(self, power_spectrogram, min_freq_bin, max_freq_bin, value, weight=1, substract=False):
        sub_power_spec = power_spectrogram.clone()
        part_of_spec = sub_power_spec[:, min_freq_bin:max_freq_bin]
        if substract:
            part_of_spec = part_of_spec - value * weight
            part_of_spec[part_of_spec < 0] = 0
        else:
            part_of_spec[part_of_spec < value * weight] = 0
        sub_power_spec[:, min_freq_bin:max_freq_bin] = part_of_spec
        return sub_power_spec

    """ Morphological operations on a certain frequency range of the original spectrogram  """
    def morph_val_from_to(self, power_spectrogram, min_freq_bin, max_freq_bin, kernel_sizex, kernel_sizey, type):
        types = {"erode" : cv2.MORPH_ERODE, "dilate" : cv2.MORPH_DILATE,  "close" : cv2.MORPH_CLOSE, "open" : cv2.MORPH_OPEN}
        respective_type = types.get(type)
        sub_power_spec = power_spectrogram.clone()
        part_of_spec = sub_power_spec[:, min_freq_bin:max_freq_bin]
        part_of_spec = torch.tensor(cv2.morphologyEx(part_of_spec.numpy(), respective_type, np.ones((kernel_sizex,kernel_sizey), np.uint8)))
        sub_power_spec[:, min_freq_bin:max_freq_bin] = part_of_spec
        return sub_power_spec

    """ Maximum filter """
    def search_maxima_spec(self, power_spectrogram, radius=3):
        power_spectrogram_max = power_spectrogram.clone().squeeze(0).transpose(0, 1).numpy()
        image_max = ndi.maximum_filter(power_spectrogram_max, size=radius, mode='nearest')
        image_max = self.np_array_to_tensor(image_max)
        return image_max

    """ Clear DC-part of a spectrogram  """
    def clear_spec_dc_part(self, power_spectrogram, up_to_bin=50):
        power_spec_clean_dc = power_spectrogram.clone().squeeze().transpose(1, 0)
        n_freq_bins = power_spec_clean_dc[0, :]
        n_freq_bins[n_freq_bins != 0] = 0
        bins_to_rem = up_to_bin
        power_spec_clean_dc[0:bins_to_rem, :] = n_freq_bins
        power_spec_clean_dc = power_spec_clean_dc.transpose(0, 1)
        return power_spec_clean_dc

    """ Clear (set spectral values to 0) spectrogram within a certain frequency range  """
    def clear_spec_min_to_max_freq(self, min_freq_bin, max_freq_bin, power_spectrogram):
        power_spec_hard_clean = power_spectrogram.clone().squeeze().transpose(1, 0)
        n_freq_bins = power_spec_hard_clean[min_freq_bin:max_freq_bin, :]
        n_freq_bins[n_freq_bins != 0] = 0
        power_spec_hard_clean[min_freq_bin:max_freq_bin, :] = n_freq_bins
        power_spec_hard_clean = power_spec_hard_clean.transpose(0, 1)
        return power_spec_hard_clean

    """ Clear spectral values smaller a certain threshold  """
    def clear_values_smaller_threshold(self, power_spectrogram, threshold=None):
        if not threshold:
            non_zero_elements_in_spec = self.get_non_zero_elements_tensor(power_spectrogram)
            sum_spec_values = power_spectrogram.sum().float()
            threshold = (sum_spec_values / non_zero_elements_in_spec)
        power_spec_2 = power_spectrogram.clone().transpose(1, 0)
        for freq_band in power_spec_2:
            freq_band[freq_band <= threshold] = 0
        power_spec_2 = power_spec_2.transpose(0, 1)
        return power_spec_2

    """ Time-based zero padding of the spectrogram """
    def pad(self, spectrogram, sequence_length, random=True):
        spectrogram = spectrogram.transpose(0, 1)
        if random:
            _sampler = lambda x: torch.randint(
                0, x, size=(1,), dtype=torch.long
            ).item()
        else:
            _sampler = lambda x: x // 2
        sample_length = spectrogram.shape[1]
        if sample_length < sequence_length:
            start = _sampler(sequence_length - sample_length)
            end = start + sample_length
            shape = list(spectrogram.shape)
            shape[1] = sequence_length
            padded_spectrogram = torch.zeros(shape, dtype=spectrogram.dtype)
            padded_spectrogram[:, start:end] = spectrogram
            padded_spectrogram = padded_spectrogram.transpose(0, 1)
            return padded_spectrogram
        else:
            spectrogram = spectrogram.transpose(0, 1)
            return spectrogram

    """ Extract spectrogram based on start and end times  """
    def extract_spec(self, spectrogram, start, end, target_len):
        if start is None and end is None:
            return self.pad(spectrogram, target_len)
        else:
            if end > spectrogram.shape[0]:
                end = spectrogram.shape[0]
            if start < 0:
                start = 0
            return torch.index_select(spectrogram, 0, torch.arange(start, end, dtype=torch.long))

    """ Identifying spectral strong intensity regions based on a give target length of the original spectrogram (sliding winodw approach, intensity curve, peak picking) """
    def identify(self, spectrogram, power_spec_max, target_len=128, perc_of_max_signal=1.0, min_bin_of_interest=100, max_bin_of_inerest=750):
        signals = dict()
        if spectrogram.shape[0] > target_len:
            start = 0
            end = start + target_len
            while end < spectrogram.shape[0]:
                spec_to_identify = torch.index_select(power_spec_max, 0, torch.arange(start, end, dtype=torch.long))
                spec_to_identify = spec_to_identify.transpose(0, 1)
                signal_strength = torch.sum(spec_to_identify[min_bin_of_interest:max_bin_of_inerest, :])
                signals[end] = signal_strength.item()
                start = start + 1
                end = start + target_len
            time = np.array(list(signals.keys()))
            signal_strengths = np.array(list(signals.values()))

            peaks, peak_dict = find_peaks(signal_strengths, distance=150, height=perc_of_max_signal * np.max(signal_strengths))

            if peaks.size > 0:
                times = []

                if perc_of_max_signal == 1.0:
                    max_idx = np.argmax(signal_strengths[peaks])
                    times.append((time[peaks[max_idx]] - target_len, time[peaks[max_idx]]))
                    return times

                for p in peaks:
                    times.append((time[p] - target_len, time[p]))

                return times

            else:
                return [(time[np.argmax(signal_strengths)] - target_len, time[np.argmax(signal_strengths)])]
        else:
            return [(None, None)]

    """ Detection algorithm of spectral strong intensity regions (vocalization areas)  """
    def detect_strong_spectral_region(self, spectrogram, spectrogram_to_extract, min_bin_of_interest=100, max_bin_of_inerest=750, exp_non_linearity=10, threshold=0.5, n_fft=4096, kernel_sizes=[9,3,5], target_len=128, perc_of_max_signal=1.0):
        spectrogram = spectrogram.squeeze(dim=0)

        spectral_bins = max_bin_of_inerest-min_bin_of_interest

        moving_avg_bin = int(0.1 * spectral_bins)
        if moving_avg_bin > 50:
            moving_avg_bin = 50

        morph_max_bin_of_interest = int(0.35 * spectral_bins)

        max_bin = math.floor(n_fft / 2) + 1

        final_spec = torch.zeros([spectrogram.shape[0], spectrogram.shape[1]])

        power_clear = self.search_maxima_spec(spectrogram, radius=kernel_sizes[1])

        power_clear = torch.tensor(exposure.equalize_hist(np.nan_to_num(power_clear.numpy())), dtype=torch.float)

        min_val = torch.min(power_clear)
        max_val = torch.max(power_clear)
        init_threshold = (min_val + max_val) / 2

        power_clear = self.clear_values_smaller_threshold(power_clear, init_threshold)

        upper_bound = max_bin
        power_clear = self.clear_spec_min_to_max_freq(max_bin_of_inerest, upper_bound, power_clear)
        power_clear = self.clear_spec_min_to_max_freq(0, min_bin_of_interest, power_clear)

        power_clear = self.insert_non_linearity(power_clear, min_freq_bin=min_bin_of_interest, max_freq_bin=max_bin_of_inerest, exponent=exp_non_linearity)

        power_clear = self.clear_values_smaller_threshold(power_spectrogram=power_clear, threshold=threshold)

        bin_range = moving_avg_bin
        for cur_pos in range(min_bin_of_interest, max_bin_of_inerest, bin_range):
            cur_spec = self.clear_spec_min_to_max_freq(cur_pos + bin_range, max_bin, power_clear)
            cur_spec = self.clear_spec_dc_part(cur_spec, up_to_bin=cur_pos)
            cur_spec = self.clear_val_from_to(cur_spec, cur_pos, cur_pos + bin_range,
                                              self.get_mean_from_to(cur_spec, cur_pos, cur_pos + bin_range), weight=1, substract=False)
            final_spec += cur_spec.clone()

        power_clear = self.morph_val_from_to(final_spec, min_bin_of_interest, morph_max_bin_of_interest, kernel_sizes[0], kernel_sizes[1], "erode")

        power_clear = torch.tensor(ndi.median_filter(power_clear.numpy(), size=kernel_sizes[2]))

        # identify spectral orca parts
        times = self.identify(spectrogram.squeeze(dim=0), power_clear, perc_of_max_signal=perc_of_max_signal, min_bin_of_interest=min_bin_of_interest, max_bin_of_inerest=max_bin_of_inerest, target_len=target_len)

        # extract spec
        target_specs = []
        if spectrogram_to_extract is not None:
            for start_t, end_t in times:
                target_specs.append(
                    self.extract_spec(spectrogram_to_extract.squeeze(dim=0), start_t, end_t, target_len))
        else:
            for start_t, end_t in times:
                target_specs.append(self.extract_spec(spectrogram, start_t, end_t, target_len))

        return target_specs, times

    """ Create binary mask out of a give spectrogram  """
    def create_mask(self, trainable_spectrogram, nfft=4096, sr=44100, fmin=800, fmax=10000, kernel_sizes=[4,3,5,2], threshold=0.025):
        trainable_spectrogram = trainable_spectrogram.squeeze()
        final_spec = torch.zeros([trainable_spectrogram.shape[0], trainable_spectrogram.shape[1]])
        max_bin = math.floor(nfft/2)+1

        lower_bound = math.ceil(fmin/((sr/2)/max_bin))
        upper_bound = math.ceil(fmax/((sr/2)/max_bin))

        spectral_bins = upper_bound - lower_bound

        moving_avg_bin = int(0.1 * spectral_bins)
        if moving_avg_bin > 25:
            moving_avg_bin = 25

        radius = kernel_sizes[0]

        power_train = trainable_spectrogram.clone()

        power_clear = self.search_maxima_spec(power_train, radius=radius)

        bin_range = moving_avg_bin
        for cur_pos in range(lower_bound, max_bin, bin_range):
            cur_spec = self.clear_spec_min_to_max_freq(cur_pos + bin_range, max_bin, power_clear)
            cur_spec = self.clear_spec_dc_part(cur_spec, up_to_bin=cur_pos)

            cur_spec = self.clear_val_from_to(cur_spec, cur_pos, cur_pos + bin_range,
                                              self.get_mean_from_to(cur_spec, cur_pos, cur_pos + bin_range), weight=1,
                                              substract=True)

            final_spec += cur_spec.clone()

        final_spec = self.clear_values_smaller_threshold(final_spec, threshold)

        if self.get_non_zero_elements_tensor(final_spec) != 0:
            power_clear = torch.tensor(self.otsu_filter(final_spec), dtype=torch.float)

        power_clear = self.morph_val_from_to(power_clear, lower_bound, upper_bound, kernel_sizes[1], kernel_sizes[1], "erode")

        power_clear = torch.tensor(ndi.median_filter(power_clear.numpy(), size=kernel_sizes[2]))

        power_clear = self.morph_val_from_to(power_clear, lower_bound, upper_bound, kernel_sizes[3], kernel_sizes[3], "erode")

        power_clear = torch.tensor(ndi.median_filter(power_clear.numpy(), size=kernel_sizes[1]))

        power_clear = self.clear_spec_min_to_max_freq(0, lower_bound, power_clear)

        power_clear = self.clear_spec_min_to_max_freq(upper_bound, max_bin, power_clear)

        return power_clear

    """ Otsu thresholding of the input spectrogram  """
    def otsu_filter(self, spectrogram):
        s_image1 = U.spec2img(spectrogram)
        s_image1 = U.flip(s_image1, dim=-1)
        s_image1 = torch.transpose(s_image1, 0, 2)
        s_image1 = np.mean(np.array(s_image1), axis=2)
        s_image1 = s_image1.T
        s_image1 = s_image1.astype(np.uint8)
        thresh = filters.threshold_otsu(s_image1)
        binary_mask = s_image1 > thresh
        binary_mask = binary_mask + 0
        return binary_mask

    #################################################################
    """ Additional functionality for plotting input spectrograms """
    #################################################################

    """
    Flip data along give dimension
    Code from https://github.com/pytorch/pytorch/issues/229
    Access Data: 06.02.2021, Last Access Date: 21.12.2021
    Changes: Modified by Christian Bergler (06.02.2021)
    """
    def flip(self, x, dim=-1):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(
            x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
        )
        return x[tuple(indices)]

    def spec2img(self, spec, normalize=True, cm=viridis_cm):
        """Converts a float spectrogram tensor to a uint8 image tensor.

        Args:
            spec: Tensor of shape [N, 1, T, F], where F are the frequency bins
                  and T are the time steps.
            normalize: Normalize over batch to range 0..1.
        """
        with torch.no_grad():
            if spec.dim() == 4:
                dim = 1
                assert spec.size(1) == 1
            elif spec.dim() <= 3:
                dim = 0
                if spec.size(0) > 1 or spec.dim() == 2:
                    spec = spec.unsqueeze(dim=0)
            else:
                raise ValueError("Unsupported spec dimension.")
            img = self.flip(spec, dim=-1)
            if normalize:
                img -= img.min()
                img /= img.max() + 1e-8
            img = img.mul(255).clamp(0, 255).long()
            img = apply_cm(img.cpu(), cm, dim=dim)
            return img.mul(255).clamp(0, 255).byte()

    def plot_spectrogram(self,
                         spectrogram,
                         output_filepath=None,
                         sr: int = 44100,
                         hop_length: int = 441,
                         fmin: int = 50,
                         fmax: int = 12500,
                         title: str = "spectrogram",
                         log=False,
                         show=True,
                         axes=None,
                         ax_title=None,
                         **kwargs
                         ):
        kwargs.setdefault("cmap", plt.cm.get_cmap("viridis"))
        kwargs.setdefault("rasterized", True)
        if isinstance(spectrogram, torch.Tensor):
            spectrogram = spectrogram.squeeze().cpu().numpy()
        spectrogram = spectrogram.T
        figsize: Tuple[int, int] = (5, 10)
        figure = plt.figure(figsize=figsize)
        figure.suptitle(title)
        if log:
            f = np.logspace(np.log2(fmin), np.log2(fmax), num=spectrogram.shape[0], base=2)
        else:
            f = np.linspace(fmin, fmax, num=spectrogram.shape[0])
        t = np.arange(0, spectrogram.shape[1]) * hop_length / sr
        if axes is None:
            axes = plt.gca()
        if ax_title is not None:
            axes.set_title(ax_title)
        img = axes.pcolormesh(t, f, spectrogram, shading="auto", **kwargs)
        figure.colorbar(img, ax=axes)
        axes.set_xlim(t[0], t[-1])
        axes.set_ylim(f[0], f[-1])
        if log:
            axes.set_yscale("symlog", basey=2)
        yaxis = axes.yaxis
        yaxis.set_major_formatter(tick.ScalarFormatter())
        xaxis = axes.xaxis
        xaxis.set_label_text("time [s]")
        if show:
            plt.show()
        self.save_plot(output_filepath)

    def save_plot(self, filepath):
        plt.savefig(filepath)
        plt.close("all")