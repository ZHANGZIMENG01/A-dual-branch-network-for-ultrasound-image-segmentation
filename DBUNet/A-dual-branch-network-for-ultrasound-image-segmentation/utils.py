import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF

import os
import math
import random
import logging
import logging.handlers

import cv2
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )



def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = config.mode, 
            factor = config.factor, 
            patience = config.patience, 
            threshold = config.threshold, 
            threshold_mode = config.threshold_mode, 
            cooldown = config.cooldown, 
            min_lr = config.min_lr, 
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler
save_counter = 0


def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None, max_save_count=5):
    global save_counter


    if save_counter >= max_save_count:
        return


    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    img = (img * 255).astype(np.uint8)


    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)


    msk_pred_bin = (msk_pred * 255).astype(np.uint8)
    msk_bin = (msk * 255).astype(np.uint8)


    contours, _ = cv2.findContours(msk_pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_contours, _ = cv2.findContours(msk_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if len(img.shape) == 2 or img.shape[2] == 1:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    cv2.drawContours(img_color, contours, -1, (255, 0, 0), 1)
    cv2.drawContours(img_color, gt_contours, -1, (255, 255, 0), 1)


    plt.figure(figsize=(7, 15))


    plt.subplot(3, 1, 1)
    plt.imshow(img_color)
    plt.title('Original Image with Predicted Contours')
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.imshow(msk, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.imshow(msk_pred, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'

    plt.savefig(save_path + str(i) + '.png')
    plt.close()

    save_counter += 1


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss

class BceDiceLoss(nn.Module):
    def __init__(self,model,device,wl2=1e-4,wb=1, wd=0.5):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.model = model
        self.device = device
        self.wb = wb
        self.wd = wd
        self.wl2 = wl2

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        l2_reg = 0
        for param in self.model.parameters():
            l2_reg += torch.norm(param.to(self.device), p=2)

        loss = self.wd * diceloss + self.wb * bceloss + self.wl2 * l2_reg
        return loss
#


class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        image,enhanceimage, mask = data
        return torch.tensor(image).permute(2,0,1), torch.tensor(enhanceimage).permute(2,0,1),torch.tensor(mask).permute(2,0,1)
       

class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w
    def __call__(self, data):
        image,enhanceimage, mask = data
        return TF.resize(image, [self.size_h, self.size_w]),TF.resize(enhanceimage, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])
       

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, enhanceimage,mask = data
        if random.random() < self.p: return TF.hflip(image),TF.hflip(enhanceimage),  TF.hflip(mask)
        else: return image, enhanceimage,mask
            

class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, enhanceimage,mask = data
        if random.random() < self.p: return TF.vflip(image),TF.vflip(enhanceimage),  TF.vflip(mask)
        else: return image, enhanceimage,mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0,360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p
    def __call__(self, data):
        image, enhanceimage,mask = data
        if random.random() < self.p: return TF.rotate(image,self.angle), TF.rotate(enhanceimage,self.angle), TF.rotate(mask,self.angle)
        else: return image,enhanceimage, mask


class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'DDTI':
            if train:
                self.mean = 58.0133
                self.std = 55.3748
            else:
                self.mean = 68.471
                self.std = 40.280
        if data_name == 'NEWDDTI':
            if train:
                self.mean =64.4012
                self.std = 41.6083
            else:
                self.mean = 79.5395
                self.std = 36.7852
        if data_name == 'SUI':
            if train:
                self.mean =58.0133
                self.std =55.3748
            else:
                self.mean = 55.94
                self.std = 55.067
        if data_name == 'BUSI':
            if train:
                self.mean =78.82
                self.std =58.236
            else:
                self.mean = 76.986
                self.std = 59.252
        if data_name == 'Bbenign':
            if train:
                self.mean = 80.56
                self.std = 58.01
            else:
                self.mean = 80.23
                self.std = 59.77
        if data_name == 'Bmalign':
            if train:
                self.mean = 74.72
                self.std = 58.54
            else:
                self.mean = 71.33
                self.std = 57.90

        elif data_name=='NEWBUSI':
            if train:
                self.mean =78.2909
                self.std = 58.4702
            else:
                self.mean = 84.2107
                self.std = 56.901
            
    def __call__(self, data):
        img, enhanceimage,msk = data
        img_normalized = (img-self.mean)/self.std
        img_normalized = ((img_normalized - np.min(img_normalized)) 
                            / (np.max(img_normalized)-np.min(img_normalized))) * 255.
        return img_normalized,enhanceimage, msk

        