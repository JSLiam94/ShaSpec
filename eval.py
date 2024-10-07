import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
import csv
sys.path.append("..")
import numpy as np
from scipy.spatial import KDTree
import nibabel as nib

from scipy.spatial.distance import cdist
import pandas as pd
import torch
import torch.nn as nn

from torch.utils import data
from DualNet_SS import DualNet_SS as DualNet
from BraTSDataSet import *
import os
from math import ceil
import nibabel as nib


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    parser = argparse.ArgumentParser(description="Shared-Specific model for 3D medical image segmentation.")

    parser.add_argument("--data_dir", type=str, default='./datalist/')
    parser.add_argument("--data_list", type=str, default='val.csv',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input_size", type=str, default='80,160,160',
                        help="Comma-separated string with depth, height and width of sub-volumnes.")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of classes to predict (ET, WT, TC).")
    parser.add_argument("--restore_from", type=str, default='snapshots/conresnet/your_checkpoint_model.pth',
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--weight_std", type=bool, default=True,
                        help="whether to use weight standarization in CONV layers.")

    parser.add_argument("--norm_cfg", type=str, default='IN')  # normalization
    parser.add_argument("--activation_cfg", type=str, default='LeakyReLU')  # activation
    parser.add_argument("--mode", type=str, default='0,1,2,3')
    return parser.parse_args()


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    deps_missing = target_size[0] - img.shape[2]
    rows_missing = target_size[1] - img.shape[3]
    cols_missing = target_size[2] - img.shape[4]
    padded_img = np.pad(img, ((0, 0), (0, 0),(0, deps_missing), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


def predict_sliding(args, net, img_list, tile_size, classes):
    image, image_res = img_list
    #interp = nn.Upsample(size=tile_size, mode='trilinear', align_corners=True)
    interp = nn.Upsample(size=tile_size, mode='trilinear', align_corners=True)
    image_size = image.shape
    overlap = 1/3

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = torch.zeros((classes, image_size[2], image_size[3], image_size[4]))
    count_predictions = torch.zeros((classes, image_size[2], image_size[3], image_size[4]))

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                y1 = int(row * strideHW)
                x1 = int(col * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                y2 = min(y1 + tile_size[1], image_size[3])
                x2 = min(x1 + tile_size[2], image_size[4])
                d1 = max(int(d2 - tile_size[0]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img_res = image_res[:, :, d1:d2, y1:y2, x1:x2]
                padded_img = pad_image(img, tile_size)
                padded_img_res = pad_image(img_res, tile_size)
                padded_prediction, _, _, _ = net(torch.from_numpy(padded_img).cuda(), val=True, mode=args.mode)
                padded_prediction = torch.sigmoid(padded_prediction)  # calc sigmoid earlier

                padded_prediction = interp(padded_prediction).cpu().data  # interp
                padded_prediction = padded_prediction[0].cpu()
                prediction = padded_prediction[0:img.shape[2],0:img.shape[3], 0:img.shape[4], :]
                count_predictions[:, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    # full_probs = torch.sigmoid(full_probs)  # calc sigmoid later

    full_probs = full_probs.numpy().transpose(1,2,3,0)
    return full_probs



def compute_hd95_single(pred, label, batch_size=1):
    if pred.size == 0 and label.size == 0:
        return 0  
    if pred.size == 0 and label.size != 0:
        return 373.13  
    if pred.size != 0 and label.size == 0:
        return 373.13  

    pred_points = np.argwhere(pred)
    label_points = np.argwhere(label)

    if pred_points.size == 0 and label_points.size == 0:
        return 0  
    if pred_points.size == 0 and label_points.size != 0:
        return 373.13  
    if pred_points.size != 0 and label_points.size == 0:
        return 373.13  

    # 使用 KDTree 加速距离计算
    tree_label = KDTree(label_points)
    distances_pred_to_label = tree_label.query(pred_points, k=1)[0]

    tree_pred = KDTree(pred_points)
    distances_label_to_pred = tree_pred.query(label_points, k=1)[0]

    # 合并距离
    all_distances = np.concatenate((distances_pred_to_label, distances_label_to_pred))

    # 计算第 95 百分位数
    hd95 = np.percentile(all_distances, 95)
    return hd95

def compute_hd95(preds, labels, batch_size=1, num_threads=4):
    """
    计算 Hausdorff Distance 95% (HD95)。

    参数:
    - preds: 预测分割图，形状为 (batch_size, height, width, depth) 或 (batch_size, height, width)
    - labels: 真实标签图，形状为 (batch_size, height, width, depth) 或 (batch_size, height, width)
    - batch_size: 分批次计算距离时的批次大小
    - num_threads: 并行计算的线程数

    返回:
    - hd95: HD95 的平均值
    """
    assert preds.shape == labels.shape, "predict & target shapes don't match"
    batch_size = preds.shape[0]
    hd95_values = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(compute_hd95_single, preds[i], labels[i], batch_size) for i in range(batch_size)]
        for future in futures:
            hd95_values.append(future.result())

    return np.mean(hd95_values)

def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target shapes don't match"
    preds = preds.astype(bool)
    labels = labels.astype(bool)
    
    # 计算前景类的 Dice 系数
    intersection = np.sum(np.logical_and(preds, labels))
    union = np.sum(preds) + np.sum(labels)
    
    if union == 0:
        return 1.0  # 如果预测和标签都是空的，Dice 系数为1
    
    return 2.0 * intersection / union

def main():
    args = get_arguments()

    # os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    d, h, w = map(int, args.input_size.split(','))

    input_size = (d, h, w)

    model = DualNet(args=args, norm_cfg=args.norm_cfg, activation_cfg=args.activation_cfg,
                    num_classes=args.num_classes, weight_std=args.weight_std, self_att=True, cross_att=False)
    model = nn.DataParallel(model)

    print('loading from checkpoint: {}'.format(args.restore_from))
    if os.path.exists(args.restore_from):
        checkpoint = torch.load(args.restore_from)
        model = checkpoint['model']
        trained_iters = checkpoint['iter']
        print("Loaded model trained for", trained_iters, "iters")
    else:
        print('File not exists in the reload path: {}'.format(args.restore_from))
        exit(0)

    model.eval()
    model.cuda()

    testloader = data.DataLoader(
        BraTSValDataSet(args.data_dir, args.data_list),
        batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    dice_ET = 0
    dice_WT = 0
    dice_TC = 0
    hd95_ET = 0
    hd95_WT = 0
    hd95_TC = 0

    results = []

    for index, batch in enumerate(testloader):
        image, image_res, label, size, name, affine = batch
        size = size[0].numpy()
        affine = affine[0].numpy()
        
        with torch.no_grad():
            output = predict_sliding(args, model, [image.numpy(), image_res.numpy()], input_size, args.num_classes)

        seg_pred_3class = np.asarray(np.around(output), dtype=np.uint8)

        seg_pred_ET = seg_pred_3class[:, :, :, 0]
        seg_pred_WT = seg_pred_3class[:, :, :, 1]
        seg_pred_TC = seg_pred_3class[:, :, :, 2]
        seg_pred = np.zeros_like(seg_pred_ET)
        seg_pred = np.where(seg_pred_WT == 1, 2, seg_pred)
        seg_pred = np.where(seg_pred_TC == 1, 1, seg_pred)
        seg_pred = np.where(seg_pred_ET == 1, 4, seg_pred)
        print(f"Processed segmentation prediction for {name}")

        seg_gt = np.asarray(label[0].numpy()[:size[0], :size[1], :size[2]], dtype=int)
        seg_gt_ET = seg_gt[0, :, :, :]
        seg_gt_WT = seg_gt[1, :, :, :]
        seg_gt_TC = seg_gt[2, :, :, :]
        
        dice_ET_i = dice_score(seg_pred_ET[None, :, :, :], seg_gt_ET[None, :, :, :])
        dice_WT_i = dice_score(seg_pred_WT[None, :, :, :], seg_gt_WT[None, :, :, :])
        dice_TC_i = dice_score(seg_pred_TC[None, :, :, :], seg_gt_TC[None, :, :, :])


        hd95_ET_i = compute_hd95(seg_pred_ET[None, :, :, :], seg_gt_ET[None, :, :, :])
        print(f"Computed HD95 for ET")
        hd95_WT_i = compute_hd95(seg_pred_WT[None, :, :, :], seg_gt_WT[None, :, :, :])
        print(f"Computed HD95 for WT")
        hd95_TC_i = compute_hd95(seg_pred_TC[None, :, :, :], seg_gt_TC[None, :, :, :])
        print(f"Computed HD95 for TC")

        print('Processing {}: Dice_ET = {:.4}, Dice_WT = {:.4}, Dice_TC = {:.4}, HD95_ET = {:.4}, HD95_WT = {:.4}, HD95_TC = {:.4}, mode = {}'.format(
            name, dice_ET_i, dice_WT_i, dice_TC_i, hd95_ET_i, hd95_WT_i, hd95_TC_i, args.mode))
        
        if dice_ET_i==0:
                dice_ET_i=1
        if dice_WT_i==0:
                dice_WT_i=1
        if dice_TC_i==0:
                dice_TC_i=1
        dice_ET += dice_ET_i
        dice_WT += dice_WT_i
        dice_TC += dice_TC_i
        hd95_ET += hd95_ET_i
        hd95_WT += hd95_WT_i
        hd95_TC += hd95_TC_i

        #seg_pred = seg_pred.transpose((1, 2, 0))
        #seg_pred = seg_pred.astype(np.int16)
        #seg_pred = nib.Nifti1Image(seg_pred, affine=affine)
        #seg_save_p = os.path.join('outputs/%s.nii.gz' % (name[0]))
        #nib.save(seg_pred, seg_save_p)

        # 将结果添加到列表中
        results.append({
            'Name': name[0],
            'mode': str(args.mode),
            'Dice_ET': dice_ET_i,
            'Dice_WT': dice_WT_i,
            'Dice_TC': dice_TC_i,
            'HD95_ET': hd95_ET_i,
            'HD95_WT': hd95_WT_i,
            'HD95_TC': hd95_TC_i
        })
            #将结果保存到 CSV 文件中
        csv_file = './results_ok.csv'

        # 定义 CSV 文件的列名
        fieldnames = ['Name', 'mode', 'Dice_ET', 'Dice_WT', 'Dice_TC', 'HD95_ET', 'HD95_WT', 'HD95_TC']

        # 检查文件是否存在
        file_exists = os.path.exists(csv_file)

        # 打开文件并追加新的记录
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # 如果文件不存在，写入表头
            
            writer.writeheader()
            
            # 写入新的记录
            for result in results:
                writer.writerow(result)

        print("Results saved to", csv_file)



    dice_ET_avg = dice_ET / (index + 1)
    dice_WT_avg = dice_WT / (index + 1)
    dice_TC_avg = dice_TC / (index + 1)
    hd95_ET_avg = hd95_ET / (index + 1)
    hd95_WT_avg = hd95_WT / (index + 1)
    hd95_TC_avg = hd95_TC / (index + 1)

    print('Average score: Dice_ET = {:.4}, Dice_WT = {:.4}, Dice_TC = {:.4}, HD95_ET = {:.4}, HD95_WT = {:.4}, HD95_TC = {:.4}'.format(
        dice_ET_avg, dice_WT_avg, dice_TC_avg, hd95_ET_avg, hd95_WT_avg, hd95_TC_avg))


    # 定义 CSV 文件路径
    averages_file = './averages_ok.csv'

    # 定义 CSV 文件的列名
    fieldnames = ['mode', 'Dice_ET_Avg', 'Dice_WT_Avg', 'Dice_TC_Avg', 'HD95_ET_Avg', 'HD95_WT_Avg', 'HD95_TC_Avg']

    # 检查文件是否存在
    file_exists = os.path.exists(averages_file)

    # 创建新的记录
    new_record = {
        'mode': str(args.mode),
        'Dice_ET_Avg': dice_ET_avg,
        'Dice_WT_Avg': dice_WT_avg,
        'Dice_TC_Avg': dice_TC_avg,
        'HD95_ET_Avg': hd95_ET_avg,
        'HD95_WT_Avg': hd95_WT_avg,
        'HD95_TC_Avg': hd95_TC_avg
    }

    # 打开文件并追加新的记录
    with open(averages_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writeheader()
        
        # 写入新的记录
        writer.writerow(new_record)

    print("Averages saved to", averages_file)

if __name__ == '__main__':
    main()
