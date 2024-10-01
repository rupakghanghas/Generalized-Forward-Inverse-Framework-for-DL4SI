
from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import errno
import os
import itertools
from torchvision.models import vgg16
import torchvision.models.vgg as vgg
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from networks import forward_network, inverse_network, iunet_network

import utils.transforms as T
import json
from matplotlib.colors import ListedColormap

print(os.getcwd())

rainbow_cmap = ListedColormap(np.load('./rainbow256.npy'))

forward_model_list = [
                      forward_network.FNO2d, 
                      forward_network.WaveformNet, 
                      forward_network.WaveformNet_V2, 
                      iunet_network.IUnetForwardModel, 
                      iunet_network.UNetForwardModel, 
                      iunet_network.IUnetForwardModel_Legacy, 
                      iunet_network.UNetForwardModel_Legacy
                    ]

inverse_model_list = [
                        inverse_network.InversionNet, 
                        iunet_network.IUnetInverseModel, 
                        iunet_network.UNetInverseModel,
                        iunet_network.IUnetInverseModel_Legacy, 
                        iunet_network.UNetInverseModel_Legacy,
                      ]
joint_model_list = [iunet_network.IUnetModel, iunet_network.JointModel, iunet_network.Decouple_IUnetModel]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_optimizer(args, model, lr):
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif args.optimizer == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-06, weight_decay=args.weight_decay) #lr = 1.0 default
    elif args.optimizer == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay) # deafult lr = 0.002
    else:
        print("Optimizer not found")
    return optimizer

def get_lr_scheduler(args, optimizer):
    # LR-Schedulers
    if args.lr_scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(args.epochs-args.start_epoch)//3, gamma=0.5)
    elif args.lr_scheduler == "LinearLR":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                         start_factor=1,
                                                         end_factor = 1e-4, 
                                                         total_iters=args.epochs-args.start_epoch)
    elif args.lr_scheduler == "ExponentialLR":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif args.lr_scheduler == "CosineAnneal":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-args.start_epoch)
    elif args.lr_scheduler == "None":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(args.epochs-args.start_epoch), gamma=1.0) #Fake LR Scheduler
    else:
        print("LR Scheduler not found")
    return lr_scheduler

def get_transforms(args, ctx):
    # label transforms
    if args.velocity_transform == "min_max":
        transform_label = T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    elif args.velocity_transform == "normalize":
        transform_label = T.Normalize(ctx['label_mean'], ctx['label_std'])
    elif args.velocity_transform == "quantile":
        transform_label = T.QuantileTransform(n_quantiles=100)
    else:
        transform_label = None

    # data transforms
    if args.amplitude_transform == "min_max":
        transform_data = T.MinMaxNormalize(ctx['data_min'], ctx['data_max'])
    elif args.amplitude_transform == "normalize":
        transform_data = T.Normalize(ctx['data_mean'], ctx['data_std'])
    elif args.amplitude_transform == "quantile":
        transform_data = T.QuantileTransform(n_quantiles=100)      
    else:
        transform_data=None
    return transform_data, transform_label

def count_parameters(model, verbose=True):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_params >= 1e6:
        num_params /= 1e6
        suffix = "M"
    elif num_params >= 1e3:
        num_params /= 1e3
        suffix = "K"
    else:
        suffix = ""
    if verbose:
        print(f"Number of trainable parameters: {num_params:.2f}{suffix}")
    return num_params

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count
    
    @property
    def sum(self):
        d = torch.tensor(list(self.deque))
        return d.sum().item()

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        if isinstance(iterable, list):
            length = max(len(x) for x in iterable)
            iterable = [x if len(x) == length else itertools.cycle(x) for x in iterable]
            iterable = zip(*iterable)
        else:
            length = len(iterable)
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(length))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj # <-- yield the batch in for loop
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (length - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, length, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, length, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


# Legacy code
class ContentLoss(nn.Module):
    def __init__(self, args):
        super(ContentLoss, self).__init__()
        names = ['l1', 'l2']
        self.loss_names = ['loss_' + n for n in names]
        for key in ['lambda_' + n for n in names]:
            setattr(self, key, getattr(args, key))
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()

    def forward(self, model, input, target):
        pred = model(input)
        loss_l1 = self.l1loss(target, pred)
        loss_l2 = self.l2loss(target, pred)
        loss = loss_l1 * self.lambda_l1 + loss_l2 * self.lambda_l2
        scope = locals()
        return loss, { k: eval(k, scope) for k in self.loss_names }


# Legacy code
class IdenticalLoss(nn.Module):
    def __init__(self, args):
        super(IdenticalLoss, self).__init__()
        names = ['id1s', 'id2s']
        self.loss_names = ['loss_' + n for n in names]
        for key in ['lambda_' + n for n in names]:
            setattr(self, key, getattr(args, key))
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()

    def forward(self, model_s2v, model_v2s, input):
        mid = model_s2v(input)
        pred = model_v2s(mid)
        cal_loss = lambda x, y: (self.l1loss(x, y), self.l2loss(x, y))
        loss_id1s, loss_id2s = cal_loss(input, pred)
        loss = loss_id1s * self.lambda_id1s + loss_id2s * self.lambda_id2s
        scope = locals()
        return loss, { k: eval(k, scope) for k in self.loss_names }

# Implemented according to H-PGNN, not useful
class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()
    
    def forward(self, pred, gt):
        return torch.mean(((pred - gt) / (torch.amax(gt, (-2, -1), keepdim=True) + 1e-5)) ** 2)


class CycleLoss(nn.Module):
    def __init__(self, args):
        super(CycleLoss, self).__init__()
        names = ['g1v', 'g2v', 'g1s', 'g2s', 'c1v', 'c2v', 'c1s', 'c2s']
        self.loss_names = ['loss_' + n for n in names]
        for key in ['lambda_' + n for n in names]:
            setattr(self, key, getattr(args, key))
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()
    
    def forward(self, data, label, pred_s=None, pred_v=None, recon_s=None, recon_v=None):
        cal_loss = lambda x, y: (self.l1loss(x, y), self.l2loss(x, y))
        loss_g1v, loss_g2v, loss_g1s, loss_g2s = [0] * 4
        if pred_v is not None:
            loss_g1v, loss_g2v = cal_loss(pred_v, label) 
        if pred_s is not None:
            loss_g1s, loss_g2s = cal_loss(pred_s, data)

        loss_c1v, loss_c2v, loss_c1s , loss_c2s = [0] * 4
        if recon_v is not None:
            loss_c1v, loss_c2v = cal_loss(recon_v, label)
        if recon_s is not None:
            loss_c1s, loss_c2s = cal_loss(recon_s, data)

        loss = loss_g1v * self.lambda_g1v + loss_g2v * self.lambda_g2v + \
            loss_g1s * self.lambda_g1s + loss_g2s * self.lambda_g2s + \
            loss_c1v * self.lambda_c1v + loss_c2v * self.lambda_c2v + \
            loss_c1s * self.lambda_c1s + loss_c2s * self.lambda_c2s
        scope = locals()
        return loss, { k: eval(k, scope) for k in self.loss_names }


# Legacy code
class _CycleLoss(nn.Module):
    def __init__(self, args):
        super(_CycleLoss, self).__init__()
        names = ['g1v', 'g2v', 'g1s', 'g2s', 'c1v', 'c2v', 'c1s', 'c2s']
        self.loss_names = ['loss_' + n for n in names]
        for key in ['lambda_' + n for n in names]:
            setattr(self, key, getattr(args, key))
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()
    
    def forward(self, data, label, pred_s=None, pred_v=None, recon_s=None, recon_v=None):
        cal_loss = lambda x, y: (self.l1loss(x, y), self.l2loss(x, y))
        loss_g1v, loss_g2v, loss_g1s, loss_g2s = [0] * 4
        if pred_v is not None and (self.lambda_g1v != 0 or self.lambda_g2v != 0):
            loss_g1v, loss_g2v = cal_loss(pred_v, label) 
        if pred_s is not None and (self.lambda_g1s != 0 or self.lambda_g2s != 0):
            loss_g1s, loss_g2s = cal_loss(pred_s, data)

        loss_c1v, loss_c2v, loss_c1s , loss_c2s = [0] * 4
        if recon_v is not None and (self.lambda_c1v != 0 or self.lambda_c2v != 0):
            loss_c1v, loss_c2v = cal_loss(recon_v, label)
        if recon_s is not None and (self.lambda_c1s != 0 or self.lambda_c2s != 0):
            loss_c1s, loss_c2s = cal_loss(recon_s, data)

        loss = loss_g1v * self.lambda_g1v + loss_g2v * self.lambda_g2v + \
            loss_g1s * self.lambda_g1s + loss_g2s * self.lambda_g2s + \
            loss_c1v * self.lambda_c1v + loss_c2v * self.lambda_c2v + \
            loss_c1s * self.lambda_c1s + loss_c2s * self.lambda_c2s
        scope = locals()
        return loss, { k: eval(k, scope) for k in self.loss_names }

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ and args.world_size > 1:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)
    
    
class Wasserstein_GP(nn.Module):
    def __init__(self, device, lambda_gp):
        super(Wasserstein_GP, self).__init__()
        self.device = device
        self.lambda_gp = lambda_gp

    def forward(self, real, fake, model):
        gradient_penalty = self.compute_gradient_penalty(model, real, fake)
        loss_real = torch.mean(model(real))
        loss_fake = torch.mean(model(fake))
        loss = -loss_real + loss_fake + gradient_penalty * self.lambda_gp
        return loss, loss_real-loss_fake, gradient_penalty

    def compute_gradient_penalty(self, model, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = model(interpolates)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(real_samples.size(0), d_interpolates.size(1)).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

# Modified from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49     
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(vgg16(pretrained=True).features[:4].eval()) # relu1_2
        blocks.append(vgg16(pretrained=True).features[4:9].eval()) # relu2_2
        blocks.append(vgg16(pretrained=True).features[9:16].eval()) # relu3_3
        blocks.append(vgg16(pretrained=True).features[16:23].eval()) # relu4_3
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()

    def forward(self, input, target, rescale=True, feature_layers=[1]):
        input = input.view(-1, 1, input.shape[-2], input.shape[-1]).repeat(1, 3, 1, 1)
        target = target.view(-1, 1, target.shape[-2], target.shape[-1]).repeat(1, 3, 1, 1)
        if rescale: # from [-1, 1] to [0, 1]
            input = input / 2 + 0.5
            target = target / 2 + 0.5
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss_l1, loss_l2 = 0.0, 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss_l1 += self.l1loss(x, y)
                loss_l2 += self.l2loss(x, y)
        return loss_l1, loss_l2


class VGG16FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()

        vgg_model = vgg.vgg16(pretrained=True)
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    
    def forward(self, x, vgg_layer_output=2):
        assert vgg_layer_output <= len(self.layer_name_mapping)
        
        count = 0
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                if count == vgg_layer_output:
                    return x
                count += 1
        return None

def cal_psnr(gt, data, max_value):
    mse = np.mean((gt - data) ** 2)
    if (mse == 0):
       return 100
    return 20 * np.log10(max_value / np.sqrt(mse))



def plot_images(num_images, dataset, model, epoch, vis_folder, device, transform_data, transform_label, plot=True, save_key="results_epoch"):
    items = np.random.choice(len(dataset), num_images)

    _, amp_true, vel_true = dataset[items]
    amp_true, vel_true = torch.tensor(amp_true).to(device), torch.tensor(vel_true).to(device)

    model = model.to(device)

    if np.any([isinstance(model, item) for item in joint_model_list]):
        # if isinstance(model, iunet_network.JointModel) and isinstance(model.forward_model, forward_network.FNO2d):
        #     amp_true = torch.einsum("ijkl->iklj", amp_true)
        #     vel_true = torch.einsum("ijkl->iklj", vel_true)
        
        amp_pred = model.forward(vel_true).detach()
        if transform_data is not None:
            amp_true_np = transform_data.inverse_transform(amp_true.detach().cpu().numpy())
            amp_pred_np = transform_data.inverse_transform(amp_pred.detach().cpu().numpy())

        vel_pred = model.inverse(amp_true).detach()
        if transform_label is not None:
            vel_true_np = transform_label.inverse_transform(vel_true.detach().cpu().numpy())
            vel_pred_np = transform_label.inverse_transform(vel_pred.detach().cpu().numpy())
        
        fig, axes = plt.subplots(num_images, 6, figsize=(20, int(3*num_images)), dpi=150)
        for i in range(num_images):
            vel = np.concatenate([vel_pred_np[i, 0], vel_true_np[i, 0]], axis=1)
            amp = np.concatenate([amp_pred_np[i, 0], amp_true_np[i, 0]], axis=1)
            
            min_vel, max_vel = vel.min(), vel.max()
            
            diff_vel = vel_pred_np[i, 0] - vel_true_np[i, 0]
            diff_amp = amp_pred_np[i, 0] - amp_true_np[i, 0]
            
            ax = axes[i, 0]
            img = ax.imshow(vel_true_np[i, 0], aspect='auto', vmin=min_vel, vmax=max_vel, cmap=rainbow_cmap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Velocity GT {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 1]
            img = ax.imshow(vel_pred_np[i, 0], aspect='auto', vmin=min_vel, vmax=max_vel, cmap=rainbow_cmap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Velocity Predicted {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 2]
            img = ax.imshow(diff_vel, aspect='auto', cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Velocity Difference {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 3]
            img = ax.imshow(amp_true_np[i, 0], aspect='auto', vmin=-1, vmax=1, cmap="seismic")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Waveform GT {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 4]
            img = ax.imshow(amp_pred_np[i, 0], aspect='auto', vmin=-1, vmax=1, cmap="seismic")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Waveform Predicted {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 5]
            img = ax.imshow(diff_amp, aspect='auto', cmap="seismic", vmin=-1, vmax=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Waveform Difference {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

    # plotting for inverse problem only
    elif np.any([isinstance(model, item) for item in inverse_model_list]):
        vel_pred = model(amp_true).detach()
        
        if transform_label is not None:
            vel_true_np = transform_label.inverse_transform(vel_true.detach().cpu().numpy())
            vel_pred_np = transform_label.inverse_transform(vel_pred.detach().cpu().numpy())
        
        fig, axes = plt.subplots(num_images, 3, figsize=(10, int(3*num_images)), dpi=150)
        for i in range(num_images):
            vel = np.concatenate([vel_pred_np[i, 0], vel_true_np[i, 0]], axis=1)
            
            min_vel, max_vel = vel.min(), vel.max()
            
            diff_vel = vel_pred_np[i, 0] - vel_true_np[i, 0]
            
            ax = axes[i, 0]
            img = ax.imshow(vel_true_np[i, 0], aspect='auto', vmin=min_vel, vmax=max_vel, cmap=rainbow_cmap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Velocity GT {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 1]
            img = ax.imshow(vel_pred_np[i, 0], aspect='auto', vmin=min_vel, vmax=max_vel, cmap=rainbow_cmap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Velocity Predicted {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 2]
            img = ax.imshow(diff_vel, aspect='auto', cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Velocity Difference {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
     

    # plotting for forward problem only
    elif np.any([isinstance(model, item) for item in forward_model_list]):
#         if isinstance(model, forward_network.FNO2d):
#             amp_true = torch.einsum("ijkl->iklj", amp_true)
#             vel_true = torch.einsum("ijkl->iklj", vel_true)
        amp_pred = model(vel_true).detach()
        if transform_data is not None:
            amp_true_np = transform_data.inverse_transform(amp_true.detach().cpu().numpy())
            amp_pred_np = transform_data.inverse_transform(amp_pred.detach().cpu().numpy())

        fig, axes = plt.subplots(num_images, 3, figsize=(10, int(3*num_images)), dpi=150)
        for i in range(num_images):
            amp = np.concatenate([amp_pred_np[i, 0], amp_true_np[i, 0]], axis=1)
            diff_amp = amp_pred_np[i, 0] - amp_true_np[i, 0]
            
            ax = axes[i, 0]
            img = ax.imshow(amp_true_np[i, 0], aspect='auto', vmin=-1, vmax=1, cmap="seismic")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Waveform GT {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 1]
            img = ax.imshow(amp_pred_np[i, 0], aspect='auto', vmin=-1, vmax=1, cmap="seismic")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Waveform Predicted {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 2]
            img = ax.imshow(diff_amp, aspect='auto', cmap="seismic", vmin=-1, vmax=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Waveform Difference {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(vis_folder, f"{save_key}_{epoch}.pdf"))
    if plot:
        plt.show()
    plt.close()


def remove_direct_arrival(amps, kmax=30, kmin=80):
    """
    amps: B, C, 1000, 70
    kmax = 30 and kmin = 80 for training
    kmax = 30 and kmin = 80 for testing
    """
    amps = amps.cpu()
    max_val, max_idx = torch.topk(amps,kmax,largest=True, dim=2)

    train_zero_amps = torch.zeros(amps.shape)
    train_zero_amps.scatter_(2, max_idx, max_val)

    train_amps_filt = amps - train_zero_amps
    
    min_val, min_idx = torch.topk(train_amps_filt,kmin,largest=False, dim=2)

    train_zero_amps = torch.zeros(amps.shape)
    train_zero_amps.scatter_(2, min_idx, min_val)

    train_amps_final = train_amps_filt - train_zero_amps
    
    return train_amps_final
    
