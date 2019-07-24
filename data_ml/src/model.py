"""
ResNet_slim: 8 layers ResCNN with temporal pooling.

"""
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib, logging, os
from . import params

from torch.autograd import Variable
from torch.utils.data import Dataset

# deal with a known bug in sklearn that pollutes stdout: https://stackoverflow.com/questions/52596204/the-imp-module-is-deprecated
with contextlib.redirect_stderr(None):
    from sklearn import metrics


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
        return F.log_softmax(self.fc1(x), dim=1), x  # dim inside log_softmax = [bs, class_num]


class VGGish(nn.Module):
    """
    PyTorch implementation of the VGGish model. (Sourced from: https://github.com/tcvrick/audioset-vggish-tensorflow-to-pytorch)

    Adapted from: https://github.com/harritaylor/torch-vggish
    The following modifications were made: (i) correction for the missing ReLU layers, (ii) correction for the
    improperly formatted data when transitioning from NHWC --> NCHW in the fully-connected layers, and (iii)
    correction for flattening in the fully-connected layers.
    """

    def __init__(self,num_classes=2):
        super(VGGish, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 24, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
        )
        self.fc_class = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x).permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(self.fc_class(x),dim=1), x


# load_model()
def get_model_or_checkpoint(model_name,model_path,num_classes=2,epoch=None,nGPU=params.N_GPU,use_cuda=True):
    if model_name=="ResNet_slim":
        model = ResNet_slim(1, 32, 32, 64, 128, 64, num_classes)
    elif "AudioSet" in model_name:
        model = VGGish()
    model = nn.DataParallel(model, device_ids=[k for k in range(nGPU)])
    if use_cuda:
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

def _unfreeze_nn_params(module):
    for p in module.parameters():
        p.requires_grad = True

def get_finetune_model(model_name,model_path,finetune_checkpoint,nGPU=params.N_GPU,use_cuda=True):
    # initialize model
    net, curr_epoch = get_model_or_checkpoint(model_name,model_path,nGPU)

    if curr_epoch == 0:
        # partially load from finetune_checkpoint 
        net.module.load_state_dict(torch.load(finetune_checkpoint),strict=False)
        print("Loaded finetune checkpoint: {}".format(finetune_checkpoint))

    # freeze everything 
    for p in net.module.parameters():
        p.requires_grad = False
    # selectively unfreeze
    if model_name == "AudioSet_fc_class":
        print("Unfreezing fc_class layer")
        _unfreeze_nn_params(net.module.fc_class)
    elif model_name == "AudioSet_fc_all":
        print("Unfreezing fc + fc_class layers")
        for m in [net.module.fc,net.module.fc_class]:
            _unfreeze_nn_params(m)

    if use_cuda:
        net = net.cuda()
    return net, curr_epoch


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

if __name__ == "__main__":
    from dataloader import AudioFileDataset
    from torch.utils.data import DataLoader
    audio_file_dataset = AudioFileDataset("../train_data/wav","../train_data/train.tsv")
    model, curr_epoch = get_finetune_model("AudioSet_fc_all","../runs","../models/pytorch_vggish.pth")

    dataloader = DataLoader(audio_file_dataset, batch_size=8, shuffle=True, drop_last=True)
    b = next(iter(dataloader))
    x = b[0].unsqueeze(1).float()

    model(x)