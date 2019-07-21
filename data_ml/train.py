# imports 
import argparse, glob, os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import DataLoader
from dataloader import AudioFileDataset
from model import ResNet_slim, AverageMeter, PredScorer, set_logger, get_model_or_checkpoint

# train() 
#TODO: Refactor a bit and remove any custom/internal references
def train(iteration, dataloader, model, optimizer, records, print_freq, epoch, batchsize,logger):
    prev = time.time()
    data_time, batch_time, losses, correct, predscorer, epoch_loss = records

    for i, (data, target) in enumerate(dataloader):

        iteration += 1

        # measure data loading time
        data_time.update(time.time() - prev)
        prev = time.time()

        # forward - data:(b x 1 x N x d), target:(b), pred:(b x C)
        data, target = data.unsqueeze(1).float().cuda(), target.cuda()  
        pred = model(data)

        # compute classification error
        pred_id = torch.argmax(pred, dim=1)
        correct.update(torch.sum(pred_id == target).item())
        predscorer.update(target,pred_id)

        # compute gradient and do SGD step
        loss = F.nll_loss(pred, target)
        losses.update(loss.data.item())
        epoch_loss.update(loss.data.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure computation time
        batch_time.update(time.time() - prev)
        prev = time.time()

        #TODO: Add tensorboard
        # print progress
        if iteration % print_freq == 0:
            print('Epoch:', epoch, '\t', 'Iter:', iteration, '\t',
                  'Data:', '%.2f' % data_time.sum, '\t',
                  'Batch:', '%.2f' % batch_time.sum, '\t',
                  'Lr:', '%.4f' % optimizer.param_groups[0]['lr'], '\t',
                  'Loss:', '%.4f' % losses.avg, '\t',
                  'Accuracy:', '%.2f' % (correct.sum*100.0/(print_freq*batchsize)), '\t',
                  'F1 Global:', '%.2f' % (predscorer.F1_global)
                  )
            predscorer.log_classification_report(logger,iteration,epoch)
            data_time.reset()
            batch_time.reset()
            losses.reset()
            correct.reset()
            predscorer.reset()
    return iteration


# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-modelPath', default=None, type=str, required=True)
    parser.add_argument('-dataPath', default=None, type=str, required=True)
    parser.add_argument('-logPath', default=None, type=str, required=True)
    # select model, lr, lr plateau params
    parser.add_argument('-model', default='ResNet', type=str, required=False)
    parser.add_argument('-lr', default=0.001, type=float, required=False)
    parser.add_argument('-lrPlateauSchedule', default="3,0.05,0.5", type=str, required=False)
    parser.add_argument('-batchSize', default=32, type=int, required=False)
    parser.add_argument('-minWindowS', default=2.0, type=float, required=False)
    parser.add_argument('-maxWindowS', default=2.0, type=float, required=False)
    parser.add_argument('--preTrainedModelDir', default=None, type=str, required=False)

    parser.add_argument('-inputDim', default=80, type=int, required=False)
    parser.add_argument('-nGPU', default=1, type=int, required=False)
    parser.add_argument('-printFreq', default=100, type=int, required=False)
    parser.add_argument('-numEpochs', default=30, type=int, required=False)
    parser.add_argument('-dataloadWorkers', default=0, type=int, required=False)
    args = parser.parse_args()

    ## initialize dataloader
    data_path = Path(args.dataPath)
    wav_dir_path, tsv_path = data_path/"wav", data_path/"train.tsv"
    mean, invstd = data_path/"mean.txt", data_path/"invstd.txt"
    audio_file_dataset = AudioFileDataset(
        wav_dir_path,tsv_path,args.minWindowS,args.maxWindowS,mean=mean,invstd=invstd
        )
    dataloader = DataLoader(audio_file_dataset, batch_size=args.batchSize, shuffle=True, drop_last=True, num_workers=args.dataloadWorkers,pin_memory=True)

    ## initialize model 
    num_classes, model_name = 2, "ResNet_slim"
    model, curr_epoch = get_model_or_checkpoint(model_name,args.modelPath,args.nGPU) 
    model.train()

    ## initialize optimizers 
    ## loop epochs, train and checkpoint
    optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    lr_plateau_schedule = [ float(p) for p in args.lrPlateauSchedule.split(',') ]
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
    patience=lr_plateau_schedule[0],threshold=lr_plateau_schedule[1],factor=lr_plateau_schedule[2])

    print_freq = args.printFreq
    records = [AverageMeter(), AverageMeter(), AverageMeter(), 
                AverageMeter(), PredScorer(), AverageMeter()]
    epoch_loss = records[-1] 

    # training
    iteration, logger = 0, set_logger(args.logPath)
    for epoch in range(curr_epoch,args.numEpochs):
        iteration = train(iteration, dataloader, model,
                            optimizer, records, print_freq, epoch, args.batchSize, logger)
        scheduler.step(epoch_loss.avg)
        message = "\n### Epoch {}, Avg loss: {} ###\n".format(epoch,epoch_loss.avg)
        logger.info(message)
        if epoch % 2 == 0:
            torch.save(model.state_dict(), args.modelPath + model_name + '_Iter_' + str(epoch))
        epoch_loss.reset()
    torch.save(model.state_dict(), args.modelPath + model_name + '_Iter_' + str(args.numEpochs))
