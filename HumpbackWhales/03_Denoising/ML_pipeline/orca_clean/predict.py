#!/usr/bin/env python3

"""
Module: predict.py
Authors: Christian Bergler
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""
import os
import argparse

import torch
import torch.nn as nn

from os import listdir
from os.path import isfile, join

from models.unet_model import UNet
from data.audiodataset import DefaultSpecDatasetOps, StridedAudioDataset, SingleAudioFolder

import data.transforms as T
import data.signal as signal

import numpy
import scipy.io.wavfile
from math import ceil, floor
from utils.logging import Logger
from collections import OrderedDict

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--debug",
    dest="debug",
    action="store_true",
    help="Print additional training and model information.",
)

parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="Path to a model.",
)

parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Path to a checkpoint. "
    "If provided the checkpoint will be used instead of the model.",
)

parser.add_argument(
    "--log_dir", type=str, default=None, help="The directory to store the logs."
)

parser.add_argument(
    "--output_dir", type=str, default=None, help="The directory to store the output."
)

parser.add_argument(
    "--sequence_len", type=float, default=2, help="Sequence length in [s]."
)


parser.add_argument(
    "--batch_size", type=int, default=1, help="The number of images per batch."
)

parser.add_argument(
    "--num_workers", type=int, default=4, help="Number of workers used in data-loading"
)

parser.add_argument(
    "--no_cuda",
    dest="cuda",
    action="store_false",
    help="Do not use cuda to train model.",
)

parser.add_argument(
    "--visualize",
    dest="visualize",
    action="store_true",
    help="Additional visualization of the noisy vs. denoised spectrogram",
)

parser.add_argument(
    "--jit_load",
    dest="jit_load",
    action="store_true",
    help="Load model via torch jit (otherwise via torch load).",
)

parser.add_argument(
    "--min_max_norm",
    dest="min_max_norm",
    action="store_true",
    help="activates min-max normalization instead of default 0/1-dB-normalization.",
)

parser.add_argument(
    "--input_file",
    type=str,
    default=None,
    help="Input file could either be a directory with multiple audio files or just one single audio file"
)

ARGS = parser.parse_args()

log = Logger("PREDICT", ARGS.debug, ARGS.log_dir)

"""
Main function to compute prediction by using a trained model together with the given input
"""
if __name__ == "__main__":

    if ARGS.checkpoint_path is not None:
        log.info(
            "Restoring checkpoint from {} instead of using a model file.".format(
                ARGS.checkpoint_path
            )
        )
        checkpoint = torch.load(ARGS.checkpoint_path)
        model = UNet(1, 1, bilinear=False)
        model.load_state_dict(checkpoint["modelState"])
        log.warning(
            "Using default preprocessing options. Provide Model file if they are changed"
        )
        dataOpts = DefaultSpecDatasetOps
    else:
        if ARGS.jit_load:
            extra_files = {}
            extra_files['dataOpts'] = ''
            model = torch.jit.load(ARGS.model_path, _extra_files=extra_files)
            unetState = model.state_dict()
            dataOpts = eval(extra_files['dataOpts'])
            log.debug("Model successfully load via torch jit: " + str(ARGS.model_path))
        else:
            model_dict = torch.load(ARGS.model_path)
            model = UNet(1, 1, bilinear=False)
            model.load_state_dict(model_dict["unetState"])
            model = nn.Sequential(
                OrderedDict([("denoiser", model)])
            )
            dataOpts = model_dict["dataOpts"]
            log.debug("Model successfully load via torch load: " + str(ARGS.model_path))

    log.info(model)

    if ARGS.visualize:
        sp = signal.signal_proc()
    else:
        sp = None

    if torch.cuda.is_available() and ARGS.cuda:
        model = model.cuda()

    model.eval()

    sr = dataOpts['sr']
    fmin = dataOpts["fmin"]
    fmax = dataOpts["fmax"]
    n_fft = dataOpts["n_fft"]
    hop_length = dataOpts["hop_length"]
    n_freq_bins = dataOpts["n_freq_bins"]
    freq_cmpr = dataOpts["freq_compression"]
    DefaultSpecDatasetOps["min_level_db"] = dataOpts["min_level_db"]
    DefaultSpecDatasetOps["ref_level_db"] = dataOpts["ref_level_db"]

    log.debug("dataOpts: " + str(dataOpts))

    if ARGS.min_max_norm:
        log.debug("Init min-max-normalization activated")
    else:
        log.debug("Init 0/1-dB-normalization activated")

    sequence_len = int(ceil(ARGS.sequence_len * sr))

    hop = sequence_len

    input_file = ARGS.input_file

    if os.path.isdir(input_file):

        log.debug("Init Single Folder Audio Dataset - Predicting Files")
        log.debug("Audio folder to process: "+str(input_file))
        audio_files = [f for f in listdir(input_file) if isfile(join(input_file, f))]

        dataset = SingleAudioFolder(
            file_names=audio_files,
            working_dir=input_file,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_freq_bins=n_freq_bins,
            freq_compression=freq_cmpr,
            f_min=fmin,
            f_max=fmax,
            center=True,
            min_max_normalize=ARGS.min_max_norm
        )

        log.info("number of files to predict={}".format(len(audio_files)))
        log.info("files will be entirely denoised without subsampling parts and/or padding")
        concatenate = False
    elif os.path.isfile(input_file):

        log.debug("Init Strided Audio Dataset - Predicting Files")
        log.debug("Audio file to process: "+str(input_file))

        dataset = StridedAudioDataset(
             input_file.strip(),
             sequence_len=sequence_len,
             hop=hop,
             sr=sr,
             fft_size=n_fft,
             fft_hop=hop_length,
             n_freq_bins=n_freq_bins,
             f_min=fmin,
             f_max=fmax,
             freq_compression=freq_cmpr,
             center=True,
             min_max_normalize=ARGS.min_max_norm
        )

        log.info("size of the file(samples)={}".format(dataset.n_frames))
        log.info("size of hop(samples)={}".format(hop))
        stop = int(max(floor(dataset.n_frames / hop), 1))
        log.info("stop time={}".format(stop))
        concatenate = True
        total_audio = None
    else:
        raise Exception("Not a valid data format - neither folder nor file")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ARGS.batch_size,
        num_workers=ARGS.num_workers,
        pin_memory=True,
    )

    t_decompr_f = T.Decompress(f_min=fmin, f_max=fmax, n_fft=n_fft, sr=sr)

    with torch.no_grad():

        for i, input in enumerate(data_loader):

            sample_spec_orig, input, spec_cmplx, filename = input

            print("current file in process, " + str(i) + "-iterations: " + str(filename[0]))

            if torch.cuda.is_available() and ARGS.cuda:
                input = input.cuda()

            denoised_output = model(input)

            decompressed_net_out = t_decompr_f(denoised_output)

            spec_cmplx = spec_cmplx.squeeze(dim=0)

            decompressed_net_out = decompressed_net_out.unsqueeze(dim=-1)

            audio_spec = decompressed_net_out * spec_cmplx

            window = torch.hann_window(n_fft)

            audio_spec = audio_spec.squeeze(dim=0).transpose(0, 1)

            detected_spec_cmplx = spec_cmplx.squeeze(dim=0).transpose(0, 1)

            if sp is not None:
                sp.plot_spectrogram(spectrogram=input.squeeze(dim=0), title="",
                                    output_filepath=ARGS.output_dir + "/net_input_spec_" + str(i) + "_" + filename[0].split("/")[-1].split(".")[0]+".pdf",
                                    sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax, show=False, ax_title="spectrogram")

                sp.plot_spectrogram(spectrogram=denoised_output.squeeze(dim=0), title="",
                                    output_filepath=ARGS.output_dir + "/net_out_spec_" + str(i) + "_" + filename[0].split("/")[-1].split(".")[0]+".pdf",
                                    sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax, show=False, ax_title="spectrogram")

            if concatenate:
                audio_out_denoised = torch.istft(audio_spec, n_fft, hop_length=hop_length, onesided=True, center=True, window=window)
                if total_audio is None:
                    total_audio = audio_out_denoised
                else:
                    total_audio = torch.cat((total_audio, audio_out_denoised), 0)
            else:
                total_audio = torch.istft(audio_spec, n_fft, hop_length=hop_length, onesided=True, center=True, window=window)
                total_audio = total_audio.numpy().T * numpy.iinfo(numpy.int16).max
                total_audio = numpy.asarray(total_audio, dtype=numpy.int16)
                scipy.io.wavfile.write(ARGS.output_dir + "/denoised_" + str(i) + "_" + filename[0].split("/")[-1].split(".")[0]+".wav", sr, total_audio)

        if concatenate:
            total_audio = total_audio.numpy().T * numpy.iinfo(numpy.int16).max
            total_audio = numpy.asarray(total_audio, dtype=numpy.int16)
            scipy.io.wavfile.write(ARGS.output_dir+"/denoised_" + str(i) + "_" + filename[0].split("/")[-1].split(".")[0]+".wav", sr, total_audio)

    log.debug("Finished proccessing")

    log.close()
