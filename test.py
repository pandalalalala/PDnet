import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import *
from tensorboardX import SummaryWriter
from utils import *
from utilsData import *
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="PDNet_Test")
parser.add_argument('--DnCNN', action="store_true", help='use DnCNN as reference?')
parser.add_argument("--features", type=int, default=64, help="Number of features")
parser.add_argument("--branches", type=int, default=2, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--net", type=str, default="net.pth", help='path of log files')
parser.add_argument("--dataPath", type=str, default="datasets", help='path of testing files')
parser.add_argument("--output", type=str, default="out_from_test", help='path of log files')
#parser.add_argument("--start_index", type=int, default=0, help="starting index of testing samples")
parser.add_argument("--mode", type=str, default="pure_denoise", help='Super-resolution (S) or denoise training (N)')
parser.add_argument("--noiseLevel", type=int, default=12, help='added noise level')
parser.add_argument('--sweep', action="store_true", help='sweep across model trained in different epochs?')
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")

opt = parser.parse_args()


def main():
    writer = SummaryWriter(opt.output)
    
    # Build model
    print('Loading model ...\n')
    net = PDNet(channels=3, branches = opt.branches)
    if opt.DnCNN:
        net = DnCNN(channels=3, num_of_layers=20)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()

    if opt.sweep:
        csvfile = "log/score_epochs"
        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for epoch in range(len(epochs)):
                model.load_state_dict(torch.load(os.path.join(opt.logdir, "net%d.pth"%epoch)))
                psnrPredict = val(model, ''.join(opt.dataPath, '/test'), 0, noiseLevel=opt.noiseLevel, mode=opt.mode)
                writer.writerow([psnrPredict])
    else:
        model.load_state_dict(torch.load(os.path.join(opt.logdir, opt.net)))
        val(model, ''.join(opt.dataPath, '/test'), 0, noiseLevel=opt.noiseLevel, mode=opt.mode, ifInhomogeneous=False)
    
if __name__ == "__main__":
    main()
