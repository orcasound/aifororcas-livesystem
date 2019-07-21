"""
ResNet_slim: 8 layers ResCNN with temporal pooling.

"""
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib, logging, os
from torch.autograd import Variable
from torch.utils.data import Dataset

# deal with a known bug in sklearn that pollutes stdout: https://stackoverflow.com/questions/52596204/the-imp-module-is-deprecated
with contextlib.redirect_stderr(None):
    from sklearn import metrics

# load_model()
def get_model_or_checkpoint(model_name,model_path,nGPU,num_classes=2,epoch=None):
    model = ResNet_slim(1, 32, 32, 64, 128, 64, num_classes)
    model = nn.DataParallel(model, device_ids=[k for k in range(nGPU)])
    model = model.cuda()

    # check for latest existing checkpoint and load
    checkpoints = sorted(
        glob.glob(os.path.join(model_path,model_name+"_*")),
        key=lambda x: int(x.rsplit('_',1)[-1]),
        reverse=True
        )
    if len(checkpoints)>0:
        # load saved model
        if epoch is None: checkpoint = checkpoints[0] 
        else:
            for ckp in checkpoints:
                if int(ckp.rsplit('_',1)[-1])==epoch:
                    checkpoint = ckp
        model.load_state_dict(torch.load(checkpoint))
        curr_epoch = int(checkpoint.rsplit('_',1)[1])
        print("Loaded checkpoint: {}".format(checkpoint))
    else:
        curr_epoch = 0
    
    return model, curr_epoch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class PredScorer(object):
    "Maintains lists of preds, targets, f1 scores, and logs classification details"

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.preds_list = []
        self.targets_list = []
        self.F1_global = 0
    
    def update(self,targets,preds):
        for pred in preds.cpu().numpy(): self.preds_list.append(pred)
        for target in targets.cpu().numpy(): self.targets_list.append(target)
        self.F1_global = metrics.f1_score(self.targets_list,self.preds_list)
        self.accuracy = metrics.accuracy_score(self.targets_list,self.preds_list)
    
    def log_classification_report(self,logger,iteration,epoch):
        cl_report = metrics.classification_report(self.targets_list,self.preds_list)
        conf_mat = metrics.confusion_matrix(self.targets_list,self.preds_list)
        logger.info(
            "### Iteration {}, Epoch {}, Accuracy {:.2f} ###\nCL Report:\n{}ConfMat:\n{}".format(iteration,epoch,self.accuracy,cl_report,conf_mat)
            )

def set_logger(log_path,log_filename="clf_log.log"):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. 
    Taken from: https://github.com/akashmjn/cs224n-gpu-that-talks/blob/master/src/utils.py

    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        if os.path.isdir(log_path):
            log_path = os.path.join(log_path,log_filename)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(logging.Formatter('%(message)s'))
        # logger.addHandler(stream_handler)

    return logger


class BasicBlock_slim(nn.Module):
    def __init__(self, planes):
        super(BasicBlock_slim, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out += residual
        out = self.relu(out)

        return out


class ResNet_slim(nn.Module):
    def __init__(self, channel_0, channel_1, channel_2, channel_3, channel_4, featureD, class_n):
        super(ResNet_slim, self).__init__()
        self.featureD = featureD
        self.convlayers = nn.Sequential(
            nn.Conv2d(channel_0, channel_1, 3, (2, 2), bias=False),
            nn.BatchNorm2d(channel_1),
            nn.ReLU(inplace=True),
            BasicBlock_slim(channel_1),
            nn.Conv2d(channel_1, channel_2, 3, (2, 2), bias=False),
            nn.BatchNorm2d(channel_2),
            nn.ReLU(inplace=True),
            BasicBlock_slim(channel_2),
            nn.Conv2d(channel_2, channel_3, 3, (2, 2), bias=False),
            nn.BatchNorm2d(channel_3),
            nn.ReLU(inplace=True),
            BasicBlock_slim(channel_3),
            nn.Conv2d(channel_3, channel_4, 3, (1, 2), bias=False),
            nn.BatchNorm2d(channel_4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_4, self.featureD, 3, (1, 2), bias=False)
        )
        self.bn = nn.BatchNorm1d(self.featureD, affine=False)
        self.fc1 = nn.utils.weight_norm(nn.Linear(self.featureD, class_n))

    def forward(self, x):
        x = self.convlayers(x)
        x = F.avg_pool2d(x, [x.size()[2], x.size()[3]], stride=1)
        x = x.view(-1, self.featureD)
        x = self.bn(x)
        return F.log_softmax(self.fc1(x), dim=1)  # dim inside log_softmax = [bs, class_num]
