import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import torch
import time
from torchinfo import summary

# 统计模型参数量的函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 统计推理时间的函数
def test_inference_time(model, input_size=(1, 3, 256, 256), device='cuda'):
    """测试模型的推理时间"""
    model.eval()

    # 创建两个输入张量：img 和 en，假设它们具有相同的维度
    img_input = torch.randn(input_size).to(device)  # 第一个输入
    en_input = torch.randn(input_size).to(device)   # 第二个输入

    with torch.no_grad():
        model.to(device)

        # 预热几次，确保 GPU 处于活跃状态
        for _ in range(10):
            _ = model(img_input, en_input)  # 传递两个输入

        # 开始计时
        start_time = time.time()
        num_runs = 100
        for _ in range(num_runs):
            _ = model(img_input, en_input)  # 传递两个输入
        end_time = time.time()

    avg_inference_time = (end_time - start_time) / num_runs
    print(f"Average inference time: {avg_inference_time * 1000:.4f} ms")

def jaccard_coefficient(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return 0  # 定义分母为零时 Jaccard 系数为零
    else:
        return intersection / union
def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, enhanceimage,targets = data
        images, enhanceimage,targets = images.cuda(non_blocking=True).float(),enhanceimage.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        if config.amp:
            with autocast():
                out = model(images,enhanceimage)
                loss = criterion(out, targets)      
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out= model(images,enhanceimage)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, en,msk = data
            img, en,msk = img.cuda(non_blocking=True).float(), en.cuda(non_blocking=True).float(),msk.cuda(non_blocking=True).float()
            out = model(img,en)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)



    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)


    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        dice = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        iou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
        recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        f1 = 2 / (float(1 / precision) + float(1 / recall)) if float(1 / precision) + float(
            1 / recall) != 0 and 1 / precision != 0 and 1 / recall != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f},dice: {dice},precision: {precision},recall:{recall},f1:{f1},iou: {iou},  accuracy: {accuracy}, specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'

        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)
    if epoch == config.epochs - 1:  # 在最后一个 epoch 时统计模型参数量和推理时间
        num_params = count_parameters(model)
        print(f"Model has {num_params} trainable parameters.")
        summary(model, (3, 256, 256))
        test_inference_time(model)


# def test_one_epoch(test_loader,
#                     model,
#                     criterion,
#                     logger,
#                     config,
#                     test_data_name=None):
#     # switch to evaluate mode
#     model.eval()
#     preds = []
#     gts = []
#     loss_list = []
#     with torch.no_grad():
#         for i, data in enumerate(tqdm(test_loader)):
#             img, msk = data
#             img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
#             out = model(img)
#             loss = criterion(out, msk)
#             loss_list.append(loss.item())
#             msk = msk.squeeze(1).cpu().detach().numpy()
#             gts.append(msk)
#             if type(out) is tuple:
#                 out = out[0]
#             out = out.squeeze(1).cpu().detach().numpy()
#             preds.append(out)
#             if i % config.save_interval == 0:
#                 save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)
#
#         preds = np.array(preds).reshape(-1)
#         gts = np.array(gts).reshape(-1)
#
#         y_pre = np.where(preds>=config.threshold, 1, 0)
#         y_true = np.where(gts>=0.5, 1, 0)
#
#         confusion = confusion_matrix(y_true, y_pre)
#         TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]
#
#         accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
#         sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
#         specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
#         f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
#         miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
#
#         if test_data_name is not None:
#             log_info = f'test_datasets_name: {test_data_name}'
#             print(log_info)
#             logger.info(log_info)
#         log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
#                 specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
#         print(log_info)
#         logger.info(log_info)
#
#     return np.mean(loss_list)
def test_one_epoch(test_loader, model, criterion, logger, config, test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img,en, msk = data
            img, en,msk = img.cuda(non_blocking=True).float(), en.cuda(non_blocking=True).float(),msk.cuda(non_blocking=True).float()
            out = model(img,en)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)

            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)




            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)


        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

        accuracy = float(TN + TP) / float(TP + TN + FP + FN) if float(TP + TN + FP + FN) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        dice = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        iou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
        recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        f1 = 2 / (float(1 / precision) + float(1 / recall)) if float(1 / precision) + float(
            1 / recall) != 0 and 1 / precision != 0 and 1 / recall != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f}, dice: {dice},iou:{iou} ,recall:{recall} precision:{precision}, f1:{f1},accuracy: {accuracy}, \
                        specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

        num_params = count_parameters(model)
        print(f"Model has {num_params} trainable parameters.")
        summary(model, input_size=[(1, 3, 256, 256), (1, 3, 256, 256)])
        test_inference_time(model)

    return np.mean(loss_list)