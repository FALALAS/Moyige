import torch
from torch.nn.modules.loss import _Loss
import numpy as np
import torch.nn.functional as F
from dataset import PrePartitionDataset
from torch.utils.data import DataLoader
import datetime, time
import torch.optim as optim
from utils import mpsnr, mssim, sam
from scipy.io import loadmat
from non_local_MEM import non_local_MEM
import argparse
import os
from torch import nn
from utils import initialize_logger
from numpy.linalg import norm



def get_args():
    args = argparse.ArgumentParser()
    # input args
    args.add_argument('--data_dir', type=str, default="/home/moyige/Graph/Graph_HSI_denoising/data")
    args.add_argument('--save_dir', type=str, default="./saved_models/case1")
    args.add_argument('--case', type=int, default=1)
    args.add_argument('--dataset_length', type=int, default=10000)
    args.add_argument('--gpu_id', type=int, default=4)
    # train args
    args.add_argument('--epochs', type=int, default=200)
    args.add_argument('--lr', type=float, default=2e-4)
    args.add_argument('--min_lr', type=float, default=1e-6)
    args.add_argument('--batch_size', type=int, default=4)
    args.add_argument('--patch_size', type=int, default=128)

    return args.parse_args()


args = get_args()
logger = initialize_logger(os.path.join(args.save_dir, "train.log"))
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
torch.cuda.set_device(args.gpu_id)


#define MSELoss
def log(*args, **kwargs):
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


# load model
net = non_local_MEM(30, 64).cuda()
net.cuda()

DDataset = PrePartitionDataset(args.data_dir, args.patch_size, args.case, length=args.dataset_length)
DLoader = DataLoader(dataset=DDataset, batch_size=args.batch_size, shuffle=True, drop_last=True)


def infer(model):
    test_mat = loadmat(os.path.join(args.data_dir, f"Noisy_test_data_CASE{args.case}.mat"))
    # arr = loadmat("../test code/test_washington.mat")['washington'].transpose(2, 0, 1)
    arr = test_mat['Img'].transpose(2, 0, 1)
    gt = torch.from_numpy(arr).unsqueeze(0)

    test_arr = torch.from_numpy(test_mat['Noisy_Img'].transpose(2, 0, 1)).unsqueeze(0)
    channels = test_arr.shape[1]
    count_mask = torch.zeros_like(test_arr)
    pred = torch.zeros_like(test_arr)
    model.eval()
    with torch.no_grad():
        for i in [t for t in range(0, channels-30, 15)]+[channels-30, ]:
            count_mask[:, i: i+30] += 1
            x = test_arr[:, i: i+30].cuda().float()
            y = model(x).detach().cpu()
            pred[:, i: i + 30] += y
    pred = pred / count_mask
    psnr = mpsnr(gt[0].numpy(), pred[0].numpy())
    return psnr


def infer_MSSIM(model):
    test_mat = loadmat(os.path.join(args.data_dir, f"Noisy_test_data_CASE{args.case}.mat"))
    # arr = loadmat("../test code/test_washington.mat")['washington'].transpose(2, 0, 1)
    arr = test_mat['Img'].transpose(2, 0, 1)
    gt = torch.from_numpy(arr).unsqueeze(0)
    test_arr = torch.from_numpy(test_mat['Noisy_Img'].transpose(2, 0, 1)).unsqueeze(0)
    channels = test_arr.shape[1]
    count_mask = torch.zeros_like(test_arr)
    pred = torch.zeros_like(test_arr)
    model.eval()
    with torch.no_grad():
        for i in [t for t in range(0, channels-30, 15)]+[channels-30, ]:
            count_mask[:, i: i+30] += 1
            x = test_arr[:, i: i+30].cuda().float()
            y = model(x).detach().cpu()
            pred[:, i: i + 30] += y
    pred = pred / count_mask
    ssim = mssim(gt[0].numpy(), pred[0].numpy())
    SAM = sam(gt[0].numpy(), pred[0].numpy())
    return ssim, SAM


# begin thr train
if __name__ == '__main__':
    logger.info('===> Building model')
    # set the decaying learning rate
    optimizer = optim.Adam(net.parameters(), lr=args.lr,)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*args.dataset_length, eta_min=args.min_lr)
    loss_fuction = nn.MSELoss().cuda()
    max_psnr, max_epoch = 0, 0
    for epoch in range(args.epochs):#
        logger.info("Decaying learning rate to %g" % scheduler.get_last_lr()[0])
        net.train()
        # Extract patches from each image
        logger.info('epochs:', epoch)
        epoch_loss = 0
        # begin train
        for step, (x, y) in enumerate(DLoader):
            batch_x, batch_x_noise = x.cuda().float(), y.cuda().float()
            optimizer.zero_grad()
            denoised_x = net(batch_x_noise)
            loss = loss_fuction(denoised_x, batch_x)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (step+1) % 500 == 0:
                logger.info(f"Epoch {epoch+1}, {step+1} / {len(DDataset) // args.batch_size}, loss: {loss.item()/args.batch_size:.5f}")
        # Verification process
        start_time = time.time()
        net.eval()
        MPSNR = infer(net)
        MSSIM, SAM = infer_MSSIM(net)
        # SAM = infer(net)
        if MPSNR > max_psnr:
            max_psnr = MPSNR
            max_epoch = epoch
            checkpoint = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict(), 'epoch': epoch, 'psnr': max_psnr}
            torch.save(checkpoint, os.path.join(args.save_dir, "best.pth"))
        logger.info(f"Test MPSNR: {np.mean(MPSNR):.4f} dB, Test MSSIM: {np.mean(MSSIM):.4f}, Test SAM: {np.mean(SAM):.4f}, Max MPSNR: {max_psnr:.4f} dB Since Epoch {max_epoch} .")
        checkpoint = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                      'epoch': epoch}
        torch.save(checkpoint, os.path.join(args.save_dir, "last.pth"))

