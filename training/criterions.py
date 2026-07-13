import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric


def Dice(y_true, y_pred, epsilon, device):
    
    y_true = y_true.to(device).float() 
    y_pred = y_pred.to(device).float()
    num_classes = y_pred.size(-1)
    smooth = torch.full((num_classes,), 1e-5, dtype=torch.float32, device=device)
    class_mask = y_true + epsilon
    
    TP = y_pred * class_mask
    FP = y_pred * (1.0 - class_mask)
    FN = (1.0 - y_pred) * class_mask

    fp_sum = FP.sum(dim=0)
    fn_sum = FN.sum(dim=0)
    
    A = fp_sum / (fp_sum + fn_sum + smooth)
    A = torch.clamp(A, min=0.2, max=0.8)
    B = 1.0 - A
    num = TP.sum(dim=0).float()
    den = num + A * fp_sum + B * fn_sum
    dice = num / (den + smooth)
    return dice

def DiceLoss(y_true, y_pred, epsilon, device):
    return 1.0 - Dice(y_true, y_pred, epsilon, device).mean()


def nu_loss_mu(u, u_noised_mu1, u_noised_mu2, mu1, mu2, d, indexes, device, threshold=4, mask=None):
    u, u_noised_mu1, u_noised_mu2 = map(
        lambda x: torch.as_tensor(x, device=device).float() if isinstance(x, (np.ndarray, torch.Tensor)) else x, 
        [u, u_noised_mu1, u_noised_mu2]
    )
    batch_size = len(indexes)
    patch_num = len(indexes[0])
    noised_loss_batch = torch.zeros((batch_size, patch_num), device=device)
    mu1_loss_batch = torch.zeros((batch_size, patch_num), device=device)
    mu2_loss_batch = torch.zeros((batch_size, patch_num), device=device)
    
    for i in range(batch_size):
        for j in range(patch_num):
            z, y, x = indexes[i][j]

            d_mask = (d[i, j] <= threshold).float()

            delta_u_noised = u_noised_mu2[i, 0, z, y, x] - u_noised_mu1[i, 0, z, y, x]
            delta_mu = torch.tensor(mu2 - mu1, device=device).float()

            delta_u_mu1 = u_noised_mu1[i, 0, z, y, x] - u[i, 0, z, y, x]
            delta_u_mu2 = u_noised_mu2[i, 0, z, y, x] - u[i, 0, z, y, x]

            if mask is not None:
                m_val = mask[i, 0, z, y, x]
                noised_loss_batch[i, j] = (delta_u_noised * delta_mu * d_mask) * m_val
                mu1_loss_batch[i, j] = (delta_u_mu1 * mu1 * d_mask) * m_val
                mu2_loss_batch[i, j] = (delta_u_mu2 * mu2 * d_mask) * m_val
            else:
                noised_loss_batch[i, j] = delta_u_noised * delta_mu * d_mask
                mu1_loss_batch[i, j] = delta_u_mu1 * mu1 * d_mask
                mu2_loss_batch[i, j] = delta_u_mu2 * mu2 * d_mask

    total_loss = torch.tensor(0.0, device=device)
    mask_noised = (noised_loss_batch < 0).float()
    if torch.sum(mask_noised) > 0:
        total_loss += torch.sum(mask_noised * noised_loss_batch) / (torch.sum(mask_noised) + 1e-8)

    mask_mu1 = (mu1_loss_batch < 0).float()
    if torch.sum(mask_mu1) > 0:
        total_loss += torch.sum(mask_mu1 * mu1_loss_batch) / (torch.sum(mask_mu1) + 1e-8)

    mask_mu2 = (mu2_loss_batch < 0).float()
    if torch.sum(mask_mu2) > 0:
        total_loss += torch.sum(mask_mu2 * mu2_loss_batch) / (torch.sum(mask_mu2) + 1e-8)

    return -total_loss


def nu_loss_d(u, u_noised_mu1, u_noised_mu2, d, indexes, device, threshold=4, mask=None):
    u, u_noised_mu1, u_noised_mu2 = map(
        lambda x: torch.as_tensor(x, device=device).float() if isinstance(x, (np.ndarray, torch.Tensor)) else x, 
        [u, u_noised_mu1, u_noised_mu2]
    )
    batch_size = len(indexes)
    patch_num = len(indexes[0])
    loss = torch.tensor(0.0, device=device)

    for i in range(batch_size):
        delta_u1 = torch.zeros(patch_num, device=device)
        delta_u2 = torch.zeros(patch_num, device=device)
        
        for j in range(patch_num):
            if d[i, j] <= threshold:
                z, y, x = indexes[i][j]
                if mask is not None and mask[i, 0, z, y, x] == 0:
                    continue
                
                delta_u1[j] = u_noised_mu1[i, 0, z, y, x] - u[i, 0, z, y, x]
                delta_u2[j] = u_noised_mu2[i, 0, z, y, x] - u[i, 0, z, y, x]
            else:
                continue
        

        d_vec = d[i, :].unsqueeze(0)
        d_diff = torch.triu(d_vec - d_vec.t(), diagonal=1)
        
        u1_vec = delta_u1.unsqueeze(0)
        delta_u1_diff = torch.triu(u1_vec - u1_vec.t(), diagonal=1)
        
        u2_vec = delta_u2.unsqueeze(0)
        delta_u2_diff = torch.triu(u2_vec - u2_vec.t(), diagonal=1)

        delta1 = d_diff * delta_u1_diff
        delta2 = d_diff * delta_u2_diff

        mask1 = (delta1 > 0).float()
        mask2 = (delta2 > 0).float()

        if torch.sum(mask1) > 0:
            loss += torch.sum(delta1 * mask1) / (torch.sum(mask1) + 1e-8)
        if torch.sum(mask2) > 0:
            loss += torch.sum(delta2 * mask2) / (torch.sum(mask2) + 1e-8)

    final_loss = loss / batch_size
    return final_loss



def nu_loss_far(u, u_noised_mu1, u_noised_mu2, distance_map, device, threshold=4, mask=None):
    # distance_map: (N, D, H, W) -> (N, 1, D, H, W)
    distance_map_tensor = torch.as_tensor(distance_map, device=device).float().unsqueeze(1)
    d_mask = (distance_map_tensor > threshold).float()
    if mask is not None:
        if mask.ndim == 5:
            d_mask = d_mask * mask
        else:
            d_mask = d_mask * mask.unsqueeze(1)
    u_sum = u + u_noised_mu1 + u_noised_mu2
    numerator = torch.sum(u_sum * d_mask)
    denominator = torch.sum(d_mask) + 1e-8
    loss = numerator / denominator

    return loss


def gradient_loss(u, g, distance_map, batch_size, device, delta=1, sample_size=None):
    
    N = batch_size
    g = g.view(N, -1).to(device)
    u = u.view(N, -1).to(device)
    if not isinstance(distance_map, torch.Tensor):
        distance_map = torch.from_numpy(distance_map).to(device)
    distance_map = distance_map.view(N, -1)

    loss = torch.tensor(0.0, device=device)
    
    for b in range(N):
        g_batch = g[b, :]
        u_batch = u[b, :]
        d_batch = distance_map[b, :]

        boundary_mask = d_batch <= delta
        g_boundary = g_batch[boundary_mask]
        u_boundary = u_batch[boundary_mask]

        del g_batch, u_batch, d_batch

        n = len(g_boundary)
        if n < 2:
            continue
        
        if sample_size is None:
            current_sample_size = min(n, 50000) 
        else:
            current_sample_size = sample_size

        # random sample
        idx_i = torch.randint(0, n, (current_sample_size,), device=device)
        idx_j = torch.randint(0, n, (current_sample_size,), device=device)

        valid_mask = idx_i != idx_j
        idx_i = idx_i[valid_mask]
        idx_j = idx_j[valid_mask]

        inner_batch_size = 10000 #TAT Based on available GPU memory
        ranking_loss = torch.tensor(0.0, device=device)
        # process batches
        for start_idx in range(0, len(idx_i), inner_batch_size):
            end_idx = min(start_idx + inner_batch_size, len(idx_i))
            b_i = idx_i[start_idx:end_idx]
            b_j = idx_j[start_idx:end_idx]

            g_diff = g_boundary[b_i] - g_boundary[b_j]
            u_diff = u_boundary[b_i] - u_boundary[b_j]

            product = g_diff * u_diff
            violation_mask = (product > 0).float()
            
            if violation_mask.sum() > 0:
                ranking_loss += torch.sum(product * violation_mask) / (violation_mask.sum() + 1e-8)
            
        loss += ranking_loss
        
    final_loss = loss / N

    return final_loss


"""
uncertainty_Dice_loss
"""
def edl_loss(y_true, alpha, num_classes, current_epoch, total_epoch, annealing_steps, device, loss_type, w_ce=1.0, w_dice=1.0, w_kl=1.0):
#def edl_loss(y_true, alpha, num_classes, device, loss_type="big"):
    
    y_true = y_true.to(device).float() 
    alpha = alpha.to(device).float()
    S = torch.sum(alpha, dim=-1, keepdim=True)
    u = num_classes / S
    prob = alpha / S
    y_pred = prob.to(device).float()
    epsilon = 1e-3
    smooth = torch.full((num_classes,), 1e-5, dtype=torch.float32, device=device)
    class_mask = y_true + epsilon
    
    loss_type = 'small'
    if loss_type == "big":
        pixel_weight = torch.clamp(u, min=0.0, max=1.0)
    elif loss_type == "small":
        pixel_weight = torch.clamp(1/u, min=0.0, max=1.0)
    elif loss_type == "normal":
        pixel_weight = 1
    
    base_TP = y_pred * class_mask
    base_FP = y_pred * (1.0 - class_mask)
    base_FN = (1.0 - y_pred) * class_mask

    TP = base_TP * pixel_weight
    FP = base_FP * pixel_weight
    FN = base_FN * pixel_weight

    fp_sum = FP.sum(dim=0)
    fn_sum = FN.sum(dim=0)
    
    A = fp_sum / (fp_sum + fn_sum + smooth)
    A = torch.clamp(A, min=0.2, max=0.8)
    B = 1.0 - A
    num = TP.sum(dim=0).float()
    den = num + A * fp_sum + B * fn_sum
    dice = num / (den + smooth)

    L_DICE = 1.0 - dice.mean()
    uncertainty_Dice_loss = torch.mean(L_DICE)
    
    return uncertainty_Dice_loss, 0, 0, 0



def KL(alp, c, device):
    assert torch.all(alp >= 1), "alp needs to be greater than or equal to 1."
    alp = alp.to(device) # [NHWD,C]
    S_alp = torch.sum(alp, dim=1, keepdim=True) # [NHWD,1]
    beta = torch.ones((1, c)).to(device) # [1,C]
    S_beta = torch.sum(beta, dim=1, keepdim=True) # [1,1]
    lnB = torch.lgamma(S_alp) - torch.sum(torch.lgamma(alp), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alp)
    dg1 = torch.digamma(alp)
    kl = torch.sum((alp - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni

    return kl


def Hausdorff_Distance(y_true, y_pred, device):
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    hd_95 = HausdorffDistanceMetric(include_background=True, reduction='none', percentile=95)
    hd_95(y_pred, y_true)
    result = hd_95.aggregate()

    return result