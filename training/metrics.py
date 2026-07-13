import torch
import math
import numpy as np
import surface_distance as surfdist
import torch.nn.functional as F

#from mindspore.nn.metrics import MeanSurfaceDistance

from medpy.metric.binary import asd
from scipy.stats import spearmanr


"""
Dice
"""

def SDice(y_true, y_pred, epsilon=1e-5, device=None):
    if device is None:
        device = y_pred.device

    y_true = y_true.to(device).float().detach()
    y_pred = y_pred.to(device).float().detach()
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    
    intersection = torch.sum(y_true_flat * y_pred_flat)
    union = torch.sum(y_true_flat) + torch.sum(y_pred_flat)
    dice_coef = (2.0 * intersection) / (union + epsilon)
    
    return dice_coef


def calculate_dice_Heart(y_true, y_pred, epsilon=1e-5, device=None, num_classes=2):
    if device is None:
        device = y_pred.device

    if y_true.shape[1] > 1:
        y_true_soft = torch.argmax(y_true, dim=1, keepdim=True)
    else:
        y_true_soft = y_true
    y_pred_soft = torch.argmax(y_pred, dim=1, keepdim=True) 

    dice = torch.zeros(num_classes - 1, device=device)
    
    for i in range(num_classes - 1):
        class_id = i + 1
        pred_mask = (y_pred_soft == class_id).float()
        true_mask = (y_true_soft == class_id).float()
        dice[i] = SDice(pred_mask, true_mask, epsilon, device)
        
    return dice


def calculate_dice_Brain(y_true, y_pred, epsilon=1e-5, device=None):
    if device is None:
        device = y_pred.device

    y_true_soft = torch.argmax(y_true, dim=1, keepdim=True)
    y_pred_soft = torch.argmax(y_pred, dim=1, keepdim=True)
    # WT
    pred_wt = (y_pred_soft != 0).float()
    true_wt = (y_true_soft != 0).float()
    dice_WT = SDice(pred_wt, true_wt, epsilon, device)
    # TC
    pred_tc = ((y_pred_soft == 1) | (y_pred_soft == 3)).float()
    true_tc = ((y_true_soft == 1) | (y_true_soft == 3)).float()
    dice_TC = SDice(pred_tc, true_tc, epsilon, device)
    # ET
    pred_et = (y_pred_soft == 3).float()
    true_et = (y_true_soft == 3).float()
    dice_ET = SDice(pred_et, true_et, epsilon, device)
    
    return dice_WT, dice_TC, dice_ET

