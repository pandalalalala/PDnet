import os
import argparse
import cv2
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import *
from loss import *
from math import log10
from utilsData import *
from utils import *

os.environ["CUDADEVICE_ORDER"] = "PCIBUS_ID"

parser = argparse.ArgumentParser(description="PDNet")
parser.add_argument('--DnCNN', action="store_true", help='use DnCNN as reference?')
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--device_ids", type=int, default=2, help="move to GPU")
parser.add_argument("--features", type=int, default=64, help="Number of features")
parser.add_argument("--branches", type=int, default=2, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=10, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--dataPath", type=str, default="OASIS1700", help='path of files to process')

parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument('--crop', action="store_true", help='crop patches?')
parser.add_argument('--cropSize', type=int, default=32, help='crop patches? training images crop size')
parser.add_argument("--noiseLevel", type=int, default=12, help="adjustable noise level")
parser.add_argument("--numOfNoisySets", type=int, default=1, help="noisy image peers")

opt = parser.parse_args()
#PDLoss_criterion = nn.MSELoss(size_average=False)

def main():
    # Load dataset
    PDLoss_criterion = PDLoss2()
    print('Loading dataset ...\n')
    start = time.time()
    datasetTr = DatasetImageWithNoise(opt.dataPath, mode='train', cropPatch=opt.crop, cropSize=opt.cropSize, stride=2, augTimes=1, noiseLevel=opt.noiseLevel, numOfNoisySets=opt.numOfNoisySets)
    datasetVal = DatasetImageWithNoise(opt.dataPath, mode='val', cropSize=opt.cropSize, stride=2, augTimes=2, noiseLevel=opt.noiseLevel, numOfNoisySets=opt.numOfNoisySets)

    loaderTr = DataLoader(dataset=datasetTr, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    end = time.time()
    print (round(end - start, 7))    
    print("# of training samples: %d\n\n" % int(len(datasetTr)))

    
    # Build model
    net = PDNet(channels=3, branches = opt.branches)
    if opt.DnCNN:
        net = DnCNN(channels=3, num_of_layers=20)
        PDLoss_criterion = nn.MSELoss(size_average=False)

    net.apply(weights_init_kaiming)

    # Move to GPU
    device_ids = range(opt.device_ids)
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    PDLoss_criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # training
    writer = SummaryWriter(opt.outf)
    step = 0

    #Upsample_4x = nn.Upsample(scale_factor=4, mode='bilinear')
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 5.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)


        # train
        for i, data in enumerate(loaderTr, 0):
            imgATr, imgBTr = data[:,0], data[:,1] 
            # let's ignore the other images at the moment
            imgDiff = imgBTr - imgATr                    
            imgATr, imgBTr, imgDiff = Variable(imgATr.cuda()), Variable(imgBTr.cuda()), Variable(imgDiff.cuda())

            
            # training step
            model.train()

            # Update denoiser network
            model.zero_grad()            
            optimizer.zero_grad()
            outRes = model(imgBTr)
            loss = PDLoss_criterion(outRes, imgDiff)/ (imgATr.size()[0]*2)
            outRes = sum(outRes)
            loss.backward()
            optimizer.step()

            model.eval()

            # results            
            imgResult = torch.clamp((imgBTr - outRes), 0., 1.)
            psnrTr = batchPSNR(imgResult, imgATr, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNRTr: %.4f" %
                (epoch+1, i+1, len(loaderTr), loss.item(), psnrTr))
            
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnrTr, step)
            step += 1
        torch.save(model.state_dict(), os.path.join(opt.outf,"net%d.pth"%epoch))
                    
        ## the end of each epoch
        model.eval()
        # validate
        psnrVal = 0
        """for k in range(len(datasetVal)):
            imgValA, imgValB = torch.unsqueeze(datasetVal[k][0], 0), torch.unsqueeze(datasetVal[k][1], 0)
            imgValA, imgValB = Variable(imgValA.cuda()), Variable(imgValB.cuda())
            with torch.no_grad():
                outVal = model(imgValB)
                outVal = sum(outVal)
            psnrVal += batchPSNR(imgValB - outVal, imgValA, 1.)


        psnrVal /= len(datasetVal)
        print("\n[epoch %d] PSNRVal: %.4f" % (epoch+1, psnrVal))
        writer.add_scalar('PSNR on validation data', psnrVal, epoch)"""
        val(model, ''.join([opt.dataPath, '/val']), 0, noiseLevel=opt.noiseLevel, ifInhomogeneous=False)
        # log the images
    
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))


if __name__ == "__main__":
    main()
