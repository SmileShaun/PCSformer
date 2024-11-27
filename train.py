import os
import argparse
import json
import torch
import torch.nn as nn
import cv2
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import robust_loss_pytorch
from collections import OrderedDict
import datetime

from utils import read_model_config, AverageMeter
from model.model import PCSformer
from loss.loss import TotalLoss
from dataloader.dataloader import PairLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(config, train_loader, network, criterion, optimizer, scaler):
    losses = AverageMeter()
    torch.cuda.empty_cache() 

    network.train()
    
    for batch in train_loader:
        haze_img = batch['source'].to(device)
        gt_img = batch['target'].to(device)

        with autocast(config['train']['no_autocast']):
            output_1, output_2 = network(haze_img)
            loss = criterion(output_1, gt_img) + criterion(output_2, gt_img)

        losses.update(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return losses   


def valid(val_loader, network):
    PSNR = AverageMeter()
    torch.cuda.empty_cache()
    
    network.eval()

    for batch in val_loader:
        haze_img = batch['source'].to(device)
        gt_img = batch['target'].to(device)

        with torch.no_grad():
            output = network(haze_img)[1].clamp_(-1, 1)

        # Compute Transmission Map PSNR
        mse_loss = F.mse_loss(
            output * 0.5 + 0.5, gt_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), gt_img.size(0))
    return PSNR.avg

if __name__ == '__main__':
    config_path = 'config.yaml'
    config = read_model_config(config_path)
    
    torch.backends.cudnn.benchmark = True
    network = PCSformer(config).to(device)
    
    # Loss Function
    criterion = TotalLoss(model_resolution=config['train']['train_patch_size']).to(device)
    
    # Optimizer
    if config['train']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=config['train']['lr'])
    elif config['train']['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=config['train']['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")
    
    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['train']['epochs'], eta_min=config['train']['lr'] * 1e-2)
    
    # Accelerate Training
    scaler = GradScaler()
       
    # Dataset
    train_dataset = PairLoader(config['train']['train_data_dir'], 'train', config['train']['train_patch_size'], config['train']['edge_decay'], config['train']['only_h_flip'])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config['train']['train_batch_size'],
                                               shuffle=True,
                                               num_workers=config['train']['num_workers'],
                                               pin_memory=True,
                                               drop_last=False)
    val_dataset = PairLoader(config['train']['val_data_dir'], 'valid', config['train']['val_patch_size'])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config['train']['val_batch_size'],
                                             num_workers=config['train']['num_workers'],
                                             pin_memory=True)
    
    current_time = datetime.datetime.now()
    file_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    weight_save_dir = os.path.join(config['train']['weight_save_dir'], file_name)
    log_save_dir = os.path.join(config['train']['log_dir'], file_name)
    os.makedirs(weight_save_dir, exist_ok=True)
    os.makedirs(log_save_dir, exist_ok=True)
    writer = SummaryWriter(log_save_dir)
    
    print('==> Start training')
    best_psnr = 0
    
    for epoch in tqdm(range(config['train']['epochs'] + 1)):
        loss = train(config, train_loader, network, criterion, optimizer, scaler)
        scheduler.step()
        writer.add_scalar('loss', loss, epoch)
        
        if epoch % config['train']['eval_freq'] == 0:
            psnr_val = valid(val_loader, network)
            if psnr_val > best_psnr:
                best_psnr = psnr_val
            
            torch.save({'state_dict': network.state_dict()},
                        os.path.join(weight_save_dir, str(epoch)+"_"+str(psnr_val)+".pth"))
            writer.add_scalar('best_psnr', best_psnr, epoch)