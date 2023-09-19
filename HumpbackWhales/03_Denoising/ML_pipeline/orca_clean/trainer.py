"""
Module: trainer.py
Authors: Christian Bergler
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import copy
import math
import time
import operator
import platform
import utils.metrics as m

import torch
import torch.nn as nn

from typing import Union
from utils.logging import Logger
from utils.summary import prepare_img
from tensorboardX import SummaryWriter
from utils.checkpoints import CheckpointHandler
from utils.early_stopping import EarlyStoppingCriterion

"""
Class which implements network training, validation and testing as well as writing checkpoints, logs, summaries, and saving the final model.
"""
class Trainer:

    """
    Initializing summary writer and checkpoint handler as well as setting required variables for training.
    """
    def __init__(
        self,
        model: nn.Module,
        logger: Logger,
        prefix: str = "",
        checkpoint_dir: Union[str, None] = None,
        summary_dir: Union[str, None] = None,
        n_summaries: int = 4,
        input_shape: tuple = None,
        start_scratch: bool = False,
    ):
        self.model = model
        self.logger = logger
        self.prefix = prefix

        self.logger.info("Init summary writer")

        if summary_dir is not None:
            run_name = prefix + "_" if prefix != "" else ""
            run_name += "{time}-{host}".format(
                time=time.strftime("%y-%m-%d-%H-%M", time.localtime()),
                host=platform.uname()[1],
            )
            summary_dir = os.path.join(summary_dir, run_name)

        self.n_summaries = n_summaries
        self.writer = SummaryWriter(summary_dir)

        if input_shape is not None:
            dummy_input = torch.rand(input_shape)
            self.logger.info("Writing graph to summary")
            self.writer.add_graph(self.model, dummy_input)

        if checkpoint_dir is not None:
            self.cp = CheckpointHandler(
                checkpoint_dir, prefix=prefix, logger=self.logger
            )
        else:
            self.cp = None

        self.start_scratch = start_scratch

    """
    Starting network training from scratch or loading existing checkpoints. The model training and validation is processed for a given
    number of epochs while storing all relevant information (metrics, summaries, logs, checkpoints) after each epoch. After the training 
    is stopped (either no improvement of the chosen validation metric for a given number of epochs, or maximum training epoch is reached)
    the model will be tested on the independent test set and saved to the selected model target directory.
    """
    def fit(
        self,
        train_loader,
        val_loader,
        test_loader,
        loss_fn,
        optimizer,
        scheduler,
        n_epochs,
        val_interval,
        patience_early_stopping,
        device,
        val_metric: Union[int, str] = "loss",
        val_metric_mode: str = "min",
        start_epoch=0,
    ):
        self.logger.info("Init model on device '{}'".format(device))
        self.model = self.model.to(device)

        best_model = copy.deepcopy(self.model.state_dict())
        best_metric = 0.0 if val_metric_mode == "max" else float("inf")

        patience_stopping = math.ceil(patience_early_stopping / val_interval)
        patience_stopping = int(max(1, patience_stopping))
        early_stopping = EarlyStoppingCriterion(
            mode=val_metric_mode, patience=patience_stopping
        )

        if not self.start_scratch and self.cp is not None:
            checkpoint = self.cp.read_latest()
            if checkpoint is not None:
                try:
                    try:
                        self.model.load_state_dict(checkpoint["modelState"])
                    except RuntimeError as e:
                        self.logger.error(
                            "Failed to restore checkpoint: "
                            "Checkpoint has different parameters"
                        )
                        self.logger.error(e)
                        raise SystemExit

                    optimizer.load_state_dict(checkpoint["trainState"]["optState"])
                    start_epoch = checkpoint["trainState"]["epoch"] + 1
                    best_metric = checkpoint["trainState"]["best_metric"]
                    best_model = checkpoint["trainState"]["best_model"]
                    early_stopping.load_state_dict(
                        checkpoint["trainState"]["earlyStopping"]
                    )
                    scheduler.load_state_dict(checkpoint["trainState"]["scheduler"])
                    self.logger.info("Resuming with epoch {}".format(start_epoch))
                except KeyError:
                    self.logger.error("Failed to restore checkpoint")
                    raise

        since = time.time()

        self.logger.info("Start training model " + self.prefix)

        try:
            if val_metric_mode == "min":
                val_comp = operator.lt
            else:
                raise Exception("validation metric mode has to be set to \"min\"")
            for epoch in range(start_epoch, n_epochs):
                self.train_epoch(
                    epoch, train_loader, loss_fn, optimizer, device
                )
                if epoch % val_interval == 0 or epoch == n_epochs - 1:
                    val_loss = self.test_epoch(
                        epoch, val_loader, loss_fn, device, phase="val"
                    )
                    if val_metric == "loss":
                        val_result = val_loss
                    else:
                        raise Exception("validation metric has to be set to \"loss\"")
                    if val_comp(val_result, best_metric):
                        best_metric = val_result
                        best_model = copy.deepcopy(self.model.state_dict())
                    self.cp.write(
                        {
                            "modelState": self.model.state_dict(),
                            "trainState": {
                            "epoch": epoch,
                            "best_metric": best_metric,
                            "best_model": best_model,
                            "optState": optimizer.state_dict(),
                            "earlyStopping": early_stopping.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            },
                        }
                    )
                    scheduler.step(val_result)
                    if early_stopping.step(val_result):
                        self.logger.info(
                            "No improvment over the last {} epochs. Stopping.".format(
                                patience_early_stopping
                            )
                        )
                        break
        except Exception:
            import traceback
            self.logger.warning(traceback.format_exc())
            self.logger.warning("Aborting...")
            self.logger.close()
            raise SystemExit

        self.model.load_state_dict(best_model)
        final_loss = self.test_epoch(0, test_loader, loss_fn, device, phase="test")
        if val_metric == "loss":
            final_metric = final_loss
        else:
            raise Exception("validation metric has to be set to \"loss\"")

        time_elapsed = time.time() - since
        self.logger.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        self.logger.info("Best val metric: {:4f}".format(best_metric))
        self.logger.info("Final test metric: {:4f}".format(final_metric))

        return self.model

    """
    Training of one epoch using pre-extracted training data, loss function, optimizer, and respective metrics
    """
    def train_epoch(self, epoch, train_loader, loss_fn, optimizer, device):
        self.logger.debug("train|{}|start".format(epoch))

        self.model.train()

        epoch_start = time.time()
        start_data_loading = epoch_start
        data_loading_time = m.Sum(torch.device("cpu"))
        epoch_loss = m.Mean(device)

        for i, (features, label) in enumerate(train_loader):

            features = features.to(device)

            ground_truth = label["ground_truth"].to(device, non_blocking=True)

            data_loading_time.update(torch.Tensor([(time.time() - start_data_loading)]))

            optimizer.zero_grad()

            denoised_output = self.model(features)

            loss = loss_fn(denoised_output, ground_truth)

            loss.backward()

            optimizer.step()

            epoch_loss.update(loss)

            start_data_loading = time.time()

            if i % 5 == 0:
                self.writer.add_image(
                    tag="train" + "/ground_truth",
                    img_tensor=prepare_img(
                        ground_truth.transpose(0, 1).squeeze(dim=0), num_images=self.n_summaries, file_names=label["file_name"],
                    ),
                    global_step=epoch,
                )
                self.writer.add_image(
                    tag="train" + "/input",
                    img_tensor=prepare_img(
                        features.transpose(0, 1).squeeze(dim=0), num_images=self.n_summaries,
                        file_names=label["file_name"],
                    ),
                    global_step=epoch,
                )
                self.writer.add_image(
                    tag="train" + "/masks_pred",
                    img_tensor=prepare_img(
                        denoised_output.transpose(0, 1).squeeze(dim=0), num_images=self.n_summaries,
                        file_names=label["file_name"],
                    ),
                    global_step=epoch,
                )

        self.write_scalar_summaries_logs(
            loss=epoch_loss.get(),
            lr=optimizer.param_groups[0]["lr"],
            epoch_time=time.time() - epoch_start,
            data_loading_time=data_loading_time.get(),
            epoch=epoch,
            phase="train",
        )

        self.writer.flush()

        return epoch_loss.get()

    """ 
    Validation/Testing using pre-extracted validation/test data, given loss function and respective metrics.
    The parameter 'phase' is used to switch between validation and test
    """
    def test_epoch(self, epoch, test_loader, loss_fn, device, phase="val"):
        self.logger.debug("{}|{}|start".format(phase, epoch))

        self.model.eval()

        with torch.no_grad():
            epoch_start = time.time()
            start_data_loading = epoch_start
            data_loading_time = m.Sum(torch.device("cpu"))
            epoch_loss = m.Mean(device)

            for i, (features, label) in enumerate(test_loader):

                features = features.to(device)

                ground_truth = label["ground_truth"].to(device, non_blocking=True)

                data_loading_time.update(torch.Tensor([(time.time() - start_data_loading)]))

                denoised_output = self.model(features)

                loss = loss_fn(denoised_output, ground_truth)

                epoch_loss.update(loss)

                if i % 5 == 0:
                    self.writer.add_image(
                        tag=phase + "/ground_truth",
                        img_tensor=prepare_img(
                            ground_truth.transpose(0, 1).squeeze(dim=0), num_images=self.n_summaries,
                            file_names=label["file_name"],
                        ),
                        global_step=epoch,
                    )
                    self.writer.add_image(
                        tag=phase + "/input",
                        img_tensor=prepare_img(
                            features.transpose(0, 1).squeeze(dim=0), num_images=self.n_summaries,
                            file_names=label["file_name"],
                        ),
                        global_step=epoch,
                    )
                    self.writer.add_image(
                        tag=phase + "/masks_pred",
                        img_tensor=prepare_img(
                            denoised_output.transpose(0, 1).squeeze(dim=0), num_images=self.n_summaries,
                            file_names=label["file_name"],
                        ),
                        global_step=epoch,
                    )

                start_data_loading = time.time()

        self.write_scalar_summaries_logs(
            loss=epoch_loss.get(),
            epoch_time=time.time() - epoch_start,
            data_loading_time=data_loading_time.get(),
            epoch=epoch,
            phase=phase,
        )

        self.writer.flush()

        return epoch_loss.get()

    """
    Writes scalar summary per partition including loss, data_loading_time, epoch time
    """
    def write_scalar_summaries_logs(
        self,
        loss: float,
        lr: float = None,
        epoch_time: float = None,
        data_loading_time: float = None,
        epoch=None,
        phase="train",
    ):
        with torch.no_grad():
            log_str = phase
            if epoch is not None:
                log_str += "|{}".format(epoch)
            self.writer.add_scalar(phase + "/epoch_loss", loss, epoch)
            log_str += "|loss:{:0.3f}".format(loss)
            if lr is not None:
                self.writer.add_scalar("lr", lr, epoch)
                log_str += "|lr:{:0.2e}".format(lr)
            if epoch_time is not None:
                self.writer.add_scalar(phase + "/time", epoch_time, epoch)
                log_str += "|t:{:0.1f}".format(epoch_time)
            if data_loading_time is not None:
                self.writer.add_scalar(
                    phase + "/data_loading_time", data_loading_time, epoch
                )
            self.logger.info(log_str)