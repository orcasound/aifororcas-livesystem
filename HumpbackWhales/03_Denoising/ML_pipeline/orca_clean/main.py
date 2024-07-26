#!/usr/bin/env python3

"""
Module: main.py
Authors: Christian Bergler
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import json
import math
import pathlib
import argparse

import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim

from data.audiodataset import (
    get_audio_files_from_dir,
    get_broken_audio_files,
    DatabaseCsvSplit,
    DefaultSpecDatasetOps,
    Dataset,
)

from trainer import Trainer
from utils.logging import Logger
from collections import OrderedDict

from models.L2Loss import L2Loss
from models.unet_model import UNet

parser = argparse.ArgumentParser()

"""
Convert string to boolean.
"""


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser.add_argument(
    "-d",
    "--debug",
    dest="debug",
    action="store_true",
    help="Log additional training and model information.",
)

parser.add_argument(
    "--data_dir",
    type=str,
    help="The path to the dataset directory.",
)

parser.add_argument(
    "--training_output",
    type=str,
    help="The target data asset where to store the outputs of the training stage",
)

parser.add_argument(
    "--cache_dir",
    type=str,
    help="The path to the cache directory.",
)

parser.add_argument(
    "--model_dir",
    type=str,
    help="The directory where the model will be stored.",
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    help="The directory where the checkpoints will be stored.",
)

parser.add_argument(
    "--log_dir", type=str, default=None, help="The directory to store the logs."
)

parser.add_argument(
    "--summary_dir",
    type=str,
    help="The directory to store the tensorboard summaries.",
)

parser.add_argument(
    "--noise_dir_train",
    type=str,
    default="",
    help="Path to a directory with noise files for training noise2noise approach using real world noise.",
)

parser.add_argument(
    "--noise_dir_val",
    type=str,
    default="",
    help="Path to a directory with noise files for validation noise2noise approach using real world noise.",
)

parser.add_argument(
    "--noise_dir_test",
    type=str,
    default="",
    help="Path to a directory with noise files for testing noise2noise approach using real world noise.",
)

parser.add_argument(
    "--start_from_scratch",
    dest="start_scratch",
    action="store_true",
    help="Start taining from scratch, i.e. do not use checkpoint to restore.",
)

parser.add_argument(
    "--jit_save",
    dest="jit_save",
    action="store_true",
    help="Save model via torch.jit save functionality.",
)

parser.add_argument(
    "--max_train_epochs",
    type=int,
    default=500,
    help="Maximum number of training epochs for the model.",
)

parser.add_argument(
    "--random_val",
    dest="random_val",
    action="store_true",
    help="Select random value intervals for noise2noise and binary mask alternatives also in validation and not only during training.",
)

parser.add_argument(
    "--epochs_per_eval",
    type=int,
    default=2,
    help="The number of batches to run in between evaluations.",
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
    "--lr",
    "--learning_rate",
    type=float,
    default=1e-5,
    help="Initial learning rate. Will get multiplied by the batch size.",
)

parser.add_argument(
    "--beta1", type=float, default=0.5, help="beta1 for the adam optimizer."
)

parser.add_argument(
    "--lr_patience_epochs",
    type=int,
    default=8,
    help="Decay the learning rate after N/epochs_per_eval epochs without any improvements on the validation set.",
)

parser.add_argument(
    "--lr_decay_factor",
    type=float,
    default=0.5,
    help="Decay factor to apply to the learning rate.",
)

parser.add_argument(
    "--early_stopping_patience_epochs",
    metavar="N",
    type=int,
    default=20,
    help="Early stopping (stop training) after N/epochs_per_eval epochs without any improvements on the validation set.",
)

parser.add_argument(
    "--filter_broken_audio",
    action="store_true",
    help="Filter files which are below a minimum loudness of 1e-3 (float32).",
)

parser.add_argument(
    "--sequence_len", type=int, default=1280, help="Sequence length in ms."
)

parser.add_argument(
    "--freq_compression",
    type=str,
    default="linear",
    help="Frequency compression to reduce GPU memory usage. "
    "Options: `'linear'` (default), '`mel`', `'mfcc'`",
)

parser.add_argument(
    "--n_freq_bins",
    type=int,
    default=256,
    help="Number of frequency bins after compression.",
)

parser.add_argument("--n_fft", type=int, default=4096, help="FFT size.")

parser.add_argument("--hop_length", type=int, default=441, help="FFT hop length.")

parser.add_argument(
    "--min_max_norm",
    dest="min_max_norm",
    action="store_true",
    help="activates min-max normalization instead of default 0/1-dB-normalization.",
)

parser.add_argument(
    "--augmentation",
    type=str2bool,
    default=True,
    help="Whether to augment the input data. "
    "Validation and test data will not be augmented.",
)

parser.add_argument(
    "--perc_of_max_signal",
    type=float,
    default=1.0,
    help="The percentage multiplied with the maximum signal strength resulting in the target height for the maximum intensity peak picking orca detection algorithm.",
)

parser.add_argument(
    "--min_thres_detect",
    type=float,
    default=0.05,
    help="Minimum/Lower value (percentage) for the embedded peak finding algorithm to calculate the minimum frequency bin (n_fft/2+1 * min_thres_detect = min_freq_bin) in order to robustly calculate signal strength within a min_freq_bin and max_freq_bin range to extract a fixed temporal context from longer vocal events. For the orca detection.",
)

parser.add_argument(
    "--max_thres_detect",
    type=float,
    default=0.40,
    help="Maximum/Upper value (percentage) for the embedded peak finding algorithm to calculate the maximum frequency bin (n_fft/2+1 * max_thres_detect = max_freq_bin) in order to robustly calculate signal strength within a min_freq_bin and max_freq_bin range to extract a fixed temporal context from longer vocal events. For the orca detection.",
)

ARGS = parser.parse_args()
ARGS.cuda = torch.cuda.is_available() and ARGS.cuda
device = torch.device("cuda") if ARGS.cuda else torch.device("cpu")

log = Logger("TRAIN", ARGS.debug, ARGS.log_dir)

"""
Get audio all audio files from the given data directory except they are broken.
"""


def get_audio_files():
    audio_files = None
    if input_data.can_load_from_csv():
        log.info("Found csv files in {}".format(ARGS.data_dir))
    else:
        log.debug("Searching for audio files in {}".format(ARGS.data_dir))
        if ARGS.filter_broken_audio:
            data_dir_ = pathlib.Path(ARGS.data_dir)
            audio_files = get_audio_files_from_dir(ARGS.data_dir)
            log.debug("Moving possibly broken audio files to .bkp:")
            broken_files = get_broken_audio_files(audio_files, ARGS.data_dir)
            for f in broken_files:
                log.debug(f)
                bkp_dir = data_dir_.joinpath(f).parent.joinpath(".bkp")
                bkp_dir.mkdir(exist_ok=True)
                f = pathlib.Path(f)
                data_dir_.joinpath(f).rename(bkp_dir.joinpath(f.name))
        audio_files = list(get_audio_files_from_dir(ARGS.data_dir))
        log.info("Found {} audio files for training.".format(len(audio_files)))
        if len(audio_files) == 0:
            log.close()
            exit(1)
    return audio_files


"""
Save the trained model and corresponding options either via torch.jit and/or torch.save.
"""


def save_model(unet, dataOpts, path, model, use_jit=False):
    unet = unet.cpu()
    unet_state_dict = unet.state_dict()
    save_dict = {
        "unetState": unet_state_dict,
        "dataOpts": dataOpts,
    }
    if not os.path.isdir(ARGS.model_dir):
        os.makedirs(ARGS.model_dir)
    if use_jit:
        example = torch.rand(1, 1, 128, 256)
        extra_files = {}
        extra_files["dataOpts"] = dataOpts.__str__()
        model = torch.jit.trace(model, example)
        torch.jit.save(model, path, _extra_files=extra_files)
        log.debug("Model successfully saved via torch jit: " + str(path))
    else:
        torch.save(save_dict, path)
        log.debug("Model successfully saved via torch save: " + str(path))


"""
Main function to compute data preprocessing, network training, evaluation, and saving.
"""
if __name__ == "__main__":
    debug = ARGS.debug
    training_output = ARGS.training_output
    data_dir = ARGS.data_dir
    cache_dir = os.path.join(training_output, ARGS.cache_dir)
    model_dir = os.path.join(training_output, ARGS.model_dir)
    checkpoint_dir = os.path.join(training_output, ARGS.checkpoint_dir)
    log_dir = os.path.join(training_output, ARGS.log_dir)
    summary_dir = os.path.join(training_output, ARGS.summary_dir)
    if not cache_dir:
        os.makedirs(cache_dir)
    if not model_dir:
        os.makedirs(model_dir)
    if not checkpoint_dir:
        os.makedirs(checkpoint_dir)
    if not log_dir:
        os.makedirs(log_dir)
    if not summary_dir:
        os.makedirs(summary_dir)
    noise_dir_train = ARGS.noise_dir_train
    noise_dir_val = ARGS.noise_dir_val
    noise_dir_test = ARGS.noise_dir_test
    start_scratch = ARGS.start_scratch
    jit_save = ARGS.jit_save
    max_train_epochs = ARGS.max_train_epochs
    random_val = ARGS.random_val
    epochs_per_eval = ARGS.epochs_per_eval
    batch_size = ARGS.batch_size
    num_workers = ARGS.num_workers
    cuda = ARGS.cuda
    lr = ARGS.lr
    beta1 = ARGS.beta1
    lr_patience_epochs = ARGS.lr_patience_epochs
    lr_decay_factor = ARGS.lr_decay_factor
    early_stopping_patience_epochs = ARGS.early_stopping_patience_epochs
    filter_broken_audio = ARGS.filter_broken_audio
    sequence_len = ARGS.sequence_len
    freq_compression = ARGS.freq_compression
    n_freq_bins = ARGS.n_freq_bins
    n_fft = ARGS.n_fft
    hop_length = ARGS.hop_length
    min_max_norm = ARGS.min_max_norm
    augmentation = ARGS.augmentation
    perc_of_max_signal = ARGS.perc_of_max_signal
    min_thres_detect = ARGS.min_thres_detect
    max_thres_detect = ARGS.max_thres_detect

    log.info(f"Log additional training and model information: {debug}")
    log.info(f"The path to the dataset directory: {data_dir}")
    log.info(f"The path to the cache directory: {cache_dir}")
    log.info(f"The directory where the model will be stored: {model_dir}")
    log.info(f"The directory where the checkpoints will be stored: {checkpoint_dir}")
    log.info(f"The directory to store the logs: {log_dir}")
    log.info(f"The directory to store the tensorboard summaries: {summary_dir}")
    log.info(
        f"Path to a directory with noise files for training noise2noise approach using real world noise: {noise_dir_train}"
    )
    log.info(
        f"Path to a directory with noise files for validation noise2noise approach using real world noise: {noise_dir_val}"
    )
    log.info(
        f"Path to a directory with noise files for testing noise2noise approach using real world noise: {noise_dir_test}"
    )
    log.info(
        f"Start taining from scratch, i.e. do not use checkpoint to restore: {start_scratch}"
    )
    log.info(f"Save model via torch.jit save functionality: {jit_save}")
    log.info(f"Maximum number of training epochs for the model: {max_train_epochs}")
    log.info(
        f"Select random value intervals for noise2noise and binary mask alternatives also in validation and not only during training: {random_val}"
    )
    log.info(f"The number of batches to run in between evaluations: {epochs_per_eval}")
    log.info(f"The number of images per batch: {batch_size}")
    log.info(f"Number of workers used in data-loading: {num_workers}")
    log.info(f"GPU support: {cuda}")
    log.info(f"Initial learning rate. Will get multiplied by the batch size: {lr}")
    log.info(f"Beta1 for the adam optimizer: {beta1}")
    log.info(
        f"Decay the learning rate after N/epochs_per_eval epochs without any improvements on the validation set: {lr_patience_epochs}"
    )
    log.info(f"Decay factor to apply to the learning rate: {lr_decay_factor}")
    log.info(
        f"Early stopping (stop training) after N/epochs_per_eval epochs without any improvements on the validation set: {early_stopping_patience_epochs}"
    )
    log.info(
        f"Filter files which are below a minimum loudness of 1e-3 (float32): {filter_broken_audio}"
    )
    log.info(f"Sequence length in ms: {sequence_len}")
    log.info(f"Frequency compression to reduce GPU memory usage: {freq_compression}")
    log.info(f"Number of frequency bins after compression: {n_freq_bins}")
    log.info(f"Spectrogram FFT size: {n_fft}")
    log.info(f"Spectrogram FFT hop length: {hop_length}")
    log.info(
        f"Activates min-max normalization instead of default 0/1-dB-normalization: {min_max_norm}"
    )
    log.info(
        f"Whether to augment the input data (during training only): {augmentation}"
    )
    log.info(
        f"The percentage multiplied with the maximum signal strength resulting in the target height for the maximum intensity peak picking orca detection algorithm: {perc_of_max_signal}"
    )
    log.info(
        f"Minimum/Lower value (percentage) for the embedded peak finding algorithm to calculate the minimum frequency bin (n_fft/2+1 * min_thres_detect = min_freq_bin) in order to robustly calculate signal strength within a min_freq_bin and max_freq_bin range to extract a fixed temporal context from longer vocal events. For the orca detection: {min_thres_detect}"
    )
    log.info(
        f"Maximum/Upper value (percentage) for the embedded peak finding algorithm to calculate the maximum frequency bin (n_fft/2+1 * max_thres_detect = max_freq_bin) in order to robustly calculate signal strength within a min_freq_bin and max_freq_bin range to extract a fixed temporal context from longer vocal events. For the orca detection: {max_thres_detect}"
    )

    dataOpts = DefaultSpecDatasetOps

    for arg, value in vars(ARGS).items():
        if arg in dataOpts and value is not None:
            dataOpts[arg] = value

    lr *= batch_size

    patience_lr = math.ceil(lr_patience_epochs / epochs_per_eval)

    patience_lr = int(max(1, patience_lr))

    log.debug("dataOpts: " + json.dumps(dataOpts, indent=4))

    sequence_len = int(
        float(sequence_len) / 1000 * dataOpts["sr"] / dataOpts["hop_length"]
    )
    log.debug("Training with sequence length: {}".format(sequence_len))
    input_shape = (batch_size, 1, dataOpts["n_freq_bins"], sequence_len)

    log.info("Setting up model")

    unet = UNet(n_channels=1, n_classes=1, bilinear=False)

    log.debug("Model: " + str(unet))
    model = nn.Sequential(OrderedDict([("unet", unet)]))

    split_fracs = {"train": 0.7, "val": 0.15, "test": 0.15}
    input_data = DatabaseCsvSplit(split_fracs, working_dir=data_dir, split_per_dir=True)

    audio_files = get_audio_files()

    noise_files_train = [str(p) for p in pathlib.Path(noise_dir_train).glob("*.wav")]
    noise_files_val = [str(p) for p in pathlib.Path(noise_dir_val).glob("*.wav")]
    noise_files_test = [str(p) for p in pathlib.Path(noise_dir_test).glob("*.wav")]

    random_val = random_val

    datasets = {
        split: Dataset(
            file_names=input_data.load(split, audio_files),
            working_dir=data_dir,
            cache_dir=cache_dir,
            sr=dataOpts["sr"],
            n_fft=dataOpts["n_fft"],
            hop_length=dataOpts["hop_length"],
            n_freq_bins=dataOpts["n_freq_bins"],
            freq_compression=dataOpts["freq_compression"],
            f_min=dataOpts["fmin"],
            f_max=dataOpts["fmax"],
            seq_len=sequence_len,
            augmentation=augmentation if split == "train" else False,
            noise_files_train=noise_files_train,
            noise_files_val=noise_files_val if split == "val" else False,
            noise_files_test=noise_files_test if split == "test" else False,
            random=True
            if split == "train" or (split == "val" and random_val)
            else False,
            dataset_name=split,
            min_max_normalize=min_max_norm,
            min_thres_detect=min_thres_detect,
            max_thres_detect=max_thres_detect,
            perc_of_max_signal=perc_of_max_signal,
        )
        for split in split_fracs.keys()
    }

    dataloaders = {
        split: torch.utils.data.DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False if split == "val" or split == "test" else True,
            pin_memory=True,
        )
        for split in split_fracs.keys()
    }

    trainer = Trainer(
        model=model,
        logger=log,
        prefix="denoiser",
        checkpoint_dir=checkpoint_dir,
        summary_dir=summary_dir,
        n_summaries=4,
        start_scratch=start_scratch,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))

    metric_mode = "min"
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=metric_mode,
        patience=patience_lr,
        factor=lr_decay_factor,
        threshold=1e-3,
        threshold_mode="abs",
    )

    L2Loss = L2Loss(reduction="sum")

    model = trainer.fit(
        dataloaders["train"],
        dataloaders["val"],
        dataloaders["test"],
        loss_fn=L2Loss,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        n_epochs=max_train_epochs,
        val_interval=epochs_per_eval,
        patience_early_stopping=early_stopping_patience_epochs,
        device=device,
        val_metric="loss",
    )

    path = os.path.join(model_dir, "ORCA-CLEAN.pk")

    save_model(unet, dataOpts, path, model, use_jit=jit_save)

    log.close()
