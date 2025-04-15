import os
import sys
import time
import datetime
import json

import torch
from torch import nn
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import Compose

from iunets import iUNet
from tqdm.notebook import tqdm
from tqdm import tqdm

from networks import iunet_network, unet, autoencoder
from dataset import FWIDataset

from utils.scheduler import WarmupMultiStepLR
import utils.transforms as T
from utils.pytorch_ssim import *
import utils.utilities as utils
from utils.config_utils import get_config_name, get_latent_dim
# from ptflops import get_model_complexity_info
step = 0



def train_one_epoch(model, vgg_model, masked_criterion, optimizer, lr_scheduler, 
                    dataloader, device, epoch, print_freq, writer):
    global step
    model.train()

    # Logger setup
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('samples/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch)

    cycle_vel_loss, cycle_amp_loss = torch.tensor([0.], device=device), torch.tensor([0.], device=device)
    vel_vgg_loss, amp_vgg_loss = torch.tensor([0.], device=device), torch.tensor([0.], device=device)
    
    # for VGG amp loss
    upsample = torch.nn.Upsample(size=(70, 70), mode="bicubic")
    
    # for warm-up scheduling of cycle loss
    cycle_loss_multiplier = 1.0
    if args.warmup_cycle_epochs > 0:
        cycle_loss_multiplier = min(1.0, epoch/args.warmup_cycle_epochs)
        print(f"Cycle Loss Multiplier: {cycle_loss_multiplier}")

    for mask, amp, vel in metric_logger.log_every(dataloader, print_freq, header):
        start_time = time.time()
        
        optimizer.zero_grad()

        mask, amp, vel = mask.to(device), amp.to(device), vel.to(device)
        identity_mask = torch.ones_like(mask)

        # amp = utils.remove_direct_arrival(amp).to(device)
        amp_pred = model.forward(vel)
        vel_pred = model.inverse(amp)
        
        if args.lambda_cycle_vel > 0:
            vel_pred2 = model.inverse(amp_pred)
            cycle_vel_loss, cycle_vel_loss_g1v, cycle_vel_loss_g2v = masked_criterion(vel_pred2, vel, identity_mask)
            cycle_vel_loss = cycle_loss_multiplier * args.lambda_cycle_vel * cycle_vel_loss

            cycle_vel_loss_g1v_val = cycle_vel_loss_g1v.item()
            cycle_vel_loss_g2v_val = cycle_vel_loss_g2v.item()

            metric_logger.update(cycle_vel_loss_g1v = cycle_vel_loss_g1v_val,
                                cycle_vel_loss_g2v = cycle_vel_loss_g2v_val)
            
            if writer:
                writer.add_scalar('cycle_vel_loss_g1v', cycle_vel_loss_g1v_val, step)
                writer.add_scalar('cycle_vel_loss_g2v', cycle_vel_loss_g2v_val, step)
        
        if args.lambda_cycle_amp > 0:
            amp_pred2 = model.forward(vel_pred)
            cycle_amp_loss, cycle_amp_loss_g1v, cycle_amp_loss_g2v = masked_criterion(amp_pred2, amp, identity_mask)
            cycle_amp_loss = cycle_loss_multiplier * args.lambda_cycle_amp * cycle_amp_loss

            cycle_amp_loss_g1v_val = cycle_amp_loss_g1v.item()
            cycle_amp_loss_g2v_val = cycle_amp_loss_g2v.item()

            metric_logger.update(cycle_amp_loss_g1v = cycle_amp_loss_g1v_val, cycle_amp_loss_g2v = cycle_amp_loss_g2v_val)
            
            if writer:
                writer.add_scalar('cycle_amp_loss_g1v', cycle_amp_loss_g1v_val, step)
                writer.add_scalar('cycle_amp_loss_g2v', cycle_amp_loss_g2v_val, step)

        vel_loss, vel_loss_g1v, vel_loss_g2v = masked_criterion(vel_pred, vel, mask)
        amp_loss, amp_loss_g1v, amp_loss_g2v = masked_criterion(amp_pred, amp, mask)

        # Calculating the perceptual loss using VGG-16 model for velocity and amplitude
        if args.lambda_vgg_vel>0:
            vgg_vel = vel.repeat(1,3,1,1)
            vgg_vel_pred = vel_pred.repeat(1,3,1,1)

            with torch.no_grad():
                vgg_features_vel = vgg_model(vgg_vel, vgg_layer_output=args.vgg_layer_output)   
            vgg_features_vel_pred = vgg_model(vgg_vel_pred, vgg_layer_output=args.vgg_layer_output)

            vel_vgg_loss, vel_vgg_loss_g1v, vel_vgg_loss_g2v = masked_criterion(vgg_features_vel, vgg_features_vel_pred, mask)

            vel_vgg_loss_g1v_val = vel_vgg_loss_g1v.item()
            vel_vgg_loss_g2v_val = vel_vgg_loss_g2v.item()

            metric_logger.update(vel_vgg_loss_g1v = vel_vgg_loss_g1v_val, vel_vgg_loss_g2v = vel_vgg_loss_g2v_val)
            
            if writer:
                writer.add_scalar('vel_vgg_loss_g1v', vel_vgg_loss_g1v_val, step)
                writer.add_scalar('vel_vgg_loss_g2v', vel_vgg_loss_g2v_val, step)

        if args.lambda_vgg_amp>0:
            # converting amplitude from Bx5x1000x70 to (B*5)x3x1000x70
            # step1: Converting amplitude from Bx5x1000x70 to (B*5)x1000x70
            vgg_amp = amp.view(-1, 1000, 70)
            # step2: Converting amplitude from Bx5x1000x70 to (B*5)x1x1000x70
            vgg_amp = vgg_amp.unsqueeze(1).repeat(1,3,1,1)
            vgg_amp = upsample(vgg_amp)
            
            C = amp.shape[1]
            vgg_amp_mask = mask.repeat(1, C).view(-1)

            vgg_amp_pred = amp_pred.view(-1, 1000, 70)
            vgg_amp_pred = vgg_amp_pred.unsqueeze(1).repeat(1,3,1,1)
            vgg_amp_pred = upsample(vgg_amp_pred)

            with torch.no_grad():
                vgg_features_amp = vgg_model(vgg_amp, vgg_layer_output=args.vgg_layer_output)
            vgg_features_amp_pred = vgg_model(vgg_amp_pred, vgg_layer_output=args.vgg_layer_output)

            amp_vgg_loss, amp_vgg_loss_g1v, amp_vgg_loss_g2v = masked_criterion(vgg_features_amp, vgg_features_amp_pred, vgg_amp_mask)

            amp_vgg_loss_g1v_val = amp_vgg_loss_g1v.item()
            amp_vgg_loss_g2v_val = amp_vgg_loss_g2v.item()

            metric_logger.update(amp_vgg_loss_g1v = amp_vgg_loss_g1v_val, 
                             amp_vgg_loss_g2v = amp_vgg_loss_g2v_val)
            
            if writer:
                writer.add_scalar('amp_vgg_loss_g1v', amp_vgg_loss_g1v_val, step)
                writer.add_scalar('amp_vgg_loss_g2v', amp_vgg_loss_g2v_val, step)

        # Calcultaing the reconstruction loss on encoder-decoder for both amp and vel        
        amp_loss_recons = 0  
        vel_loss_recons = 0 
        if args.lambda_recons>0:
            vel_recons = model.vel_model.forward(vel)
            amp_recons = model.amp_model.forward(amp)
            amp_loss_recons = nn.MSELoss()(amp_recons, amp)  # Compute amplitude loss
            vel_loss_recons = nn.MSELoss()(vel_recons, vel)  # Compute velocity loss
            
            metric_logger.update(amp_loss_recons = amp_loss_recons, 
                             vel_loss_recons = vel_loss_recons)
            
            if writer:
                writer.add_scalar('amp_loss_recons', amp_loss_recons, step)
                writer.add_scalar('vel_loss_recons', vel_loss_recons, step)

        
 
        loss = args.lambda_amp * amp_loss + args.lambda_vel * vel_loss + args.lambda_vgg_vel * vel_vgg_loss + args.lambda_vgg_amp * amp_vgg_loss + cycle_vel_loss + cycle_amp_loss + args.lambda_recons * amp_loss_recons + args.lambda_recons * vel_loss_recons

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        vel_loss_g1v_val = vel_loss_g1v.item()
        vel_loss_g2v_val = vel_loss_g2v.item()
        
        amp_loss_g1v_val = amp_loss_g1v.item()
        amp_loss_g2v_val = amp_loss_g2v.item()

        batch_size = amp.shape[0]
        metric_logger.update(
                             loss = loss_val, 
                             lr = optimizer.param_groups[0]['lr'],
                             vel_loss_g1v = vel_loss_g1v_val,
                             vel_loss_g2v = vel_loss_g2v_val, 
                             amp_loss_g1v = amp_loss_g1v_val,
                             amp_loss_g2v = amp_loss_g2v_val,
                            )
        
        metric_logger.meters['samples/s'].update(batch_size / (time.time() - start_time))
        
        if writer:
            writer.add_scalar('loss', loss_val, step)
            writer.add_scalar('vel_loss_g1v', vel_loss_g1v_val, step)
            writer.add_scalar('vel_loss_g2v', vel_loss_g2v_val, step)
            writer.add_scalar('amp_loss_g1v', amp_loss_g1v_val, step)
            writer.add_scalar('amp_loss_g2v', amp_loss_g2v_val, step)

        step += 1

def evaluate(model, criterion, dataloader, device, ctx, transform_data=None, transform_label=None):
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        total_samples = 0
        
        eval_metrics = ["vel_sum_abs_error", "vel_sum_squared_error", "amp_sum_abs_error", "amp_sum_squared_error"]
        eval_dict = {}
        for metric in eval_metrics:
            eval_dict[metric] = 0

        val_loss = 0
        for _, amp, vel in dataloader:
            
            amp = amp.to(device, non_blocking=True)
            vel = vel.to(device, non_blocking=True)

            batch_size = amp.shape[0]
            total_samples += batch_size
            
            vel_pred = model.inverse(amp)
            amp_pred = model.forward(vel)

            vel_loss, vel_loss_g1v, vel_loss_g2v = criterion(vel_pred, vel)
            amp_loss, amp_loss_g1v, amp_loss_g2v = criterion(amp_pred, amp)

            loss = vel_loss + amp_loss
            val_loss += loss.item()

            eval_dict["vel_sum_abs_error"] += (l1_loss(vel, vel_pred) * batch_size).item()
            eval_dict["vel_sum_squared_error"] += (l2_loss(vel, vel_pred) * batch_size).item()
            
            eval_dict["amp_sum_abs_error"] += (l1_loss(amp, amp_pred) * batch_size).item()
            eval_dict["amp_sum_squared_error"] += (l2_loss(amp, amp_pred) * batch_size).item()

        for metric in eval_metrics:
            eval_dict[metric] /= total_samples 

    val_loss /= len(dataloader)
    return val_loss, eval_dict

def main(args):
    global step

    print(args)
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)
    
    utils.init_distributed_mode(args) # distributed mode initialization
    
    # Create path for saving
    utils.mkdir(args.output_path) # create folder to store checkpoints

    # folders to save intermediate results during training and validation
    train_vis_folder = os.path.join(args.output_path, args.plot_directory, 'train')
    valid_vis_folder = os.path.join(args.output_path, args.plot_directory, 'validation')

    if os.path.exists(train_vis_folder) == False:
        os.makedirs(train_vis_folder)

    if os.path.exists(valid_vis_folder) == False:
        os.makedirs(valid_vis_folder)

    # Set up tensorboard summary writer
    train_writer, val_writer = None, None
    if args.tensorboard:
        utils.mkdir(args.log_path) # create folder to store tensorboard logs
        if not args.distributed or (args.rank == 0) and (args.local_rank == 0):
            train_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'train'))
            val_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'val'))
                                                                 

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[args.dataset]
        except KeyError:
            print('Unsupported dataset.')
            sys.exit()

    if args.file_size is not None:
        ctx['file_size'] = args.file_size

    transform_data, transform_label = utils.get_transforms(args, ctx)

    # Create dataset and dataloader
    print('Loading data')
    print('Loading training data')
    
    if args.train_anno[-3:] == 'txt':
        dataset_train = FWIDataset(
            args.train_anno,
            preload=True,
            sample_ratio=args.sample_temporal,
            file_size=ctx['file_size'],
            transform_data=transform_data,
            transform_label=transform_label,
            mask_factor=args.mask_factor
        )
    else:
        dataset_train = torch.load(args.train_anno)

    print('Loading validation data')
    if args.val_anno[-3:] == 'txt':
        dataset_valid = FWIDataset(
            args.val_anno,
            preload=True,
            sample_ratio=args.sample_temporal,
            file_size=ctx['file_size'],
            transform_data=transform_data,
            transform_label=transform_label
        )
    else:
        dataset_valid = torch.load(args.val_anno)

    print('Creating data loaders')
    if args.distributed:
        train_sampler = DistributedSampler(dataset_train, shuffle=True)
        valid_sampler = DistributedSampler(dataset_valid, shuffle=True)
    else:
        train_sampler = RandomSampler(dataset_train)
        valid_sampler = RandomSampler(dataset_valid)

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=True, drop_last=True, collate_fn=default_collate)

    dataloader_valid = DataLoader(
        dataset_valid, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate)

    print('Creating model')
    # if args.model not in network.model_dict:
    #     print('Unsupported model.')
    #     sys.exit()

    # creating amplitude cnn
    amp_cfg_name = get_config_name(args.latent_dim, model_type="amplitude")
    amp_model = autoencoder.AutoEncoder(args.cfg_path, amp_cfg_name).to(device)

    # creating velocity cnn
    vel_cfg_name = get_config_name(args.latent_dim, model_type="velocity")
    vel_model = autoencoder.AutoEncoder(args.cfg_path, vel_cfg_name).to(device)
    
    latent_channels = get_latent_dim(args.cfg_path, amp_cfg_name)
#     print("args model: (before if statement)",args.model)
    if args.model == "IUNET":
        iunet_model = iUNet(in_channels=latent_channels, dim=2, architecture=(4,4,4,4))
        model = iunet_network.IUnetModel(amp_model, vel_model, iunet_model).to(device)
        print("IUnet model initialized.")
    
    elif args.model == "Decouple_IUnet":
        amp_iunet_model = iUNet(in_channels=latent_channels, dim=2, architecture=(4,4,4,4))
        vel_iunet_model = iUNet(in_channels=latent_channels, dim=2, architecture=(4,4,4,4))
        model = iunet_network.Decouple_IUnetModel(amp_model, vel_model, amp_iunet_model, vel_iunet_model).to(device)
        print("Decoupled IUnetModel model initialized.")
        
#     else:
#         print(f"Invalid Model: {args.model}")

    print(utils.count_parameters(model))

#     macs, params = get_model_complexity_info(model, input_res=(1, 70, 70), as_strings=True, print_per_layer_stat=True, verbose=True, ignore_modules=[torch.nn.Conv2d])
#     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    

    vgg_model = utils.VGG16FeatureExtractor().to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Define loss function
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()

    def masked_criterion(pred, gt, mask):
        B, C, H, W = pred.shape
        mask = mask.view(B, 1, 1, 1)
        num_elements = mask.sum() + 1

        squared_diff = ((pred - gt)**2) * mask
        abs_diff = (pred-gt).abs() * mask
        norm_l2_loss = torch.sum(squared_diff.mean(dim=[1, 2, 3])/num_elements)
        norm_l1_loss = torch.sum(abs_diff.mean(dim=[1, 2, 3])/num_elements)
        loss = args.lambda_g1v * norm_l1_loss + args.lambda_g2v * norm_l2_loss
        return loss, norm_l1_loss, norm_l2_loss

    def criterion(pred, gt):
        loss_g1v = l1loss(pred, gt)
        loss_g2v = l2loss(pred, gt)
        loss = args.lambda_g1v * loss_g1v + args.lambda_g2v * loss_g2v
        return loss, loss_g1v, loss_g2v

    def relative_l2_error(pred, gt):
        batch_size = gt.shape[0]
        pred = pred.view(batch_size, -1)
        gt = gt.view(batch_size, -1)

        numerator = torch.linalg.norm(pred - gt, ord=2, dim=1)
        denominator = torch.linalg.norm(gt, ord=2, dim=1)
        relative_loss = (numerator/denominator).mean()
        return relative_loss

    # Scale lr according to effective batch size
    lr = args.lr * args.world_size
    optimizer = utils.get_optimizer(args, model, lr)
    lr_scheduler = utils.get_lr_scheduler(args, optimizer)
    
    # Convert scheduler to be per iteration instead of per epoch
    warmup_iters = args.lr_warmup_epochs * len(dataloader_train)
    lr_milestones = [len(dataloader_train) * m for m in args.lr_milestones]

    model_without_ddp = model
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model']))
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        step = checkpoint['step']
        lr_scheduler.milestones = lr_milestones

    print('Start training')
    start_time = time.time()
    best_loss = np.inf
    chp = 1 

    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_one_epoch(model, vgg_model, masked_criterion, optimizer, lr_scheduler, dataloader_train,
                        device, epoch, args.print_freq, train_writer)
        lr_scheduler.step()    
       
        loss, eval_dict = evaluate(model, criterion, dataloader_valid, device, ctx, transform_data, transform_label)
        print("Test Metrics:", eval_dict)

        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'step': step,
            'args': args}

        if epoch%args.plot_interval == 0:
            utils.plot_images(args.num_images, dataset_train, model, epoch, train_vis_folder, device, transform_data, transform_label)
            utils.plot_images(args.num_images, dataset_valid, model, epoch, valid_vis_folder, device, transform_data, transform_label)

            utils.save_on_master(checkpoint, os.path.join(args.output_path, 'latest_checkpoint.pth'))

        # Save checkpoint per epoch
        if loss < best_loss:
            utils.save_on_master(checkpoint, os.path.join(args.output_path, 'checkpoint.pth'))
            print('saving checkpoint at epoch: ', epoch)
            chp = epoch
            best_loss = loss
            
        # Save checkpoint every epoch block
        print('current best loss: ', best_loss)
        print('current best epoch: ', chp)
        if args.output_path and (epoch + 1) % args.epoch_block == 0:
            utils.save_on_master(checkpoint, os.path.join(args.output_path, 'model_{}.pth'.format(epoch + 1)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Training')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='flatfault-b', type=str, help='dataset name')
    parser.add_argument('-fs', '--file-size', default=None, type=str, help='number of samples in each npy file')

    # Path related
    parser.add_argument('-ap', '--anno-path', default='split_files', help='annotation files location')
    parser.add_argument('-t', '--train-anno', default='flatfault_b_train_invnet.txt', help='name of train anno')
    parser.add_argument('-v', '--val-anno', default='flatfault_b_val_invnet.txt', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='Invnet_models', help='path to parent folder to save checkpoints')
    parser.add_argument('-l', '--log-path', default='Invnet_models', help='path to parent folder to save logs')
    parser.add_argument('-n', '--save-name', default='fcn_l1loss_ffb', help='folder name for this experiment')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')
    parser.add_argument('--plot_directory', type=str, default='visualisation', help='directory to save intermediate model results')

    # Model related
    parser.add_argument('-m', '--model', type=str, default='IUnet', help='Model Name choices: IUnet, Decouple_IUnet')
    parser.add_argument('--cfg-path', type=str, default='./configs/', help='directory for config path')
    parser.add_argument('--latent-dim', default=70, type=int)
    parser.add_argument('-um', '--up-mode', default=None, help='upsampling layer mode such as "nearest", "bicubic", etc.')
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')
    parser.add_argument('--optimizer', type=str, default="Adam", help='Optimizer')
    parser.add_argument('--lr_scheduler', type=str, default="StepLR", help='LR_Scheduler')

    # Training related
    parser.add_argument('-b', '--batch-size', default=50, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('-lm', '--lr-milestones', nargs='+', default=[], type=int, help='decrease lr on milestones')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4 , type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='number of warmup epochs')   
    parser.add_argument('-eb', '--epoch_block', type=int, default=40, help='epochs in a saved block')
    parser.add_argument('-nb', '--num_block', type=int, default=3, help='number of saved block')
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('-r', '--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--plot_interval', default=10, type=int, help='result plot frequency')
    parser.add_argument('--num_images', default=10, type=int, help='plotting 10 random images')
    parser.add_argument('--rm_direct_arrival', default=1, type=int, help='Remove direct arrival from amplitude data.')
    parser.add_argument('--velocity_transform', type=str, default="min_max", help='Transform scheme')
    parser.add_argument('--amplitude_transform', type=str, default="normalize", help='Transform scheme')
    parser.add_argument('--mask_factor', type=float, default=0.0, help=" Mask data fraction.")

    # Loss related
    parser.add_argument('-g1v', '--lambda_g1v', type=float, default=1.0)
    parser.add_argument('-g2v', '--lambda_g2v', type=float, default=1.0)
    parser.add_argument('--lambda_vel', type=float, default=1.0)
    parser.add_argument('--lambda_amp', type=float, default=1.0)
    parser.add_argument('--lambda_vgg_vel', type=float, default=0.1)
    parser.add_argument('--lambda_vgg_amp', type=float, default=0.1)
    parser.add_argument('--lambda_cycle_vel', type=float, default=0.0)
    parser.add_argument('--lambda_cycle_amp', type=float, default=0.0)
    parser.add_argument('--warmup_cycle_epochs', type=int, default=-1, help='Default value of -1 means constant cycle loss.')
    parser.add_argument('--vgg_layer_output', type=int, default=2, help='VGG16 pretrained model layer output for perceptual loss calculation.')
    parser.add_argument('--lambda_reg', type=float, default=0.1, help='lambda coefficient for TV Norm regularization.')
    parser.add_argument('--lambda_recons', type=float, default=0.0, help='lambda coefficient for reconstruction loss for both amp and vel')
    
    # Distributed training related
    parser.add_argument('--sync-bn', action='store_true', help='Use sync batch norm')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # Tensorboard related
    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard for logging.')

    args = parser.parse_args()

    # default_path = "/projects/ml4science/openfwi/openfwi_results/"
    default_path = "/projects/ml4science/OpenFWI/Results/"
    # default_path = "/globalscratch/OpenFWI/Results"

    args.output_path = os.path.join(default_path, args.output_path, args.save_name, args.suffix or '')
    args.log_path = os.path.join(default_path, args.log_path, args.save_name, args.suffix or '')
    args.train_anno = os.path.join(args.anno_path, args.train_anno)
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    
    args.epochs = args.epoch_block * args.num_block

    # if args.resume:
    #     args.resume = os.path.join(args.output_path, args.resume)

    return args

if __name__ == '__main__':
    args = parse_args()
    # utils.set_seed(1234)
    utils.set_seed(3333)
    main(args)
