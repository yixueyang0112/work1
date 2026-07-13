import torch
import gc
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from training.criterions import nu_loss_d, nu_loss_mu, nu_loss_far, gradient_loss
from utilities.utils import compute_distance_map, generate_noisy_images_Rician, sample_class_wise_noised_voxel_patch_images_Rician, run_forward_to_get_u_3d, compute_feature_gradient
from training.metrics import calculate_dice_Heart, calculate_dice_Brain



L_gradient = gradient_loss
L_noise_mu = nu_loss_mu
L_noise_d = nu_loss_d
L_noise_far = nu_loss_far
visualization_in_writer = False
binary_threshold = 0.3
default_mu = 1.0
default_d = 1.0
default_far = 1.0
annealing_hsd = False


def train(model,
          dataloader,
          optimizer,
          num_classes,
          criterion,
          current_epoch,
          total_epoch,
          annealing_steps,
          device,
          loss_type,
          batch_size,
          writer,
          good_model_step,
          num_patch,
          d_threshold,
          d_eps,
          epsilon,
          beta=0.0, 
          gamma=0.0,
          use_grad_clip=False,
          sample_size=None,
          visualization_in_writer=False,
          dataset=None,
          weight_CE=1.0,
          weight_Dice=1.0,
          weight_KL=1.0,
          **kwargs):   
    
    model.train()
    running_loss = 0.0
    running_dice_loss = 0.0
    running_kl_loss = 0.0
    running_grad_loss = 0.0
    running_noise_loss = 0.0
    running_noise_loss_d = 0.0
    running_noise_loss_mu = 0.0
    running_noise_loss_far = 0.0
    running_ce_loss = 0.0

    if 'Heart' in dataset:
        running_dice_LA = 0.0
        k = 3
    elif 'Brain' in dataset:
        running_dice_WT = 0.0
        running_dice_TC = 0.0
        running_dice_ET = 0.0
        k = 3

    C = num_classes

    for batch_idx, (images, labels, spacing) in enumerate(dataloader):
        # images: [N, 1, D, H, W], labels: [N, C, D, H, W]
        images, labels = images.to(device), labels.to(device)
        N, _, D, H, W = images.size()
        targets = labels.permute(0, 2, 3, 4, 1).contiguous().view(-1, C) 
        optimizer.zero_grad()
        
        #pred_raw = model(images) # [N, C, D, H, W]
        pred_raw, encoder_feat = model(images)
        pred = pred_raw.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        
        evidence = F.softplus(pred)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        u = (num_classes / S).view(N, D, H, W) # 恢复 3D 形状用于后续损失
        prob_flat = alpha / S
        
       
        edl_loss, term_ace, kl_loss, dice_loss = criterion(
            targets, alpha, num_classes=C, current_epoch=current_epoch, 
            total_epoch=total_epoch, annealing_steps=annealing_steps, 
            device=device, loss_type=loss_type,
            w_ce=weight_CE,
            w_dice=weight_Dice,
            w_kl=weight_KL
        )
        
        #edl_loss = criterion(targets, alpha, num_classes=C, device=device, loss_type="big")

        prob_5d = prob_flat.view(N, D, H, W, C).permute(0, 4, 1, 2, 3)
        if 'Heart' in dataset:
            dice_vals = calculate_dice_Heart(labels, prob_5d, num_classes=C)
            running_dice_LA += dice_vals[0].item()
        elif 'Brain' in dataset:
            d_wt, d_tc, d_et = calculate_dice_Brain(labels, prob_5d)
            running_dice_WT += d_wt.item()
            running_dice_TC += d_tc.item()
            running_dice_ET += d_et.item()
        del pred, evidence, alpha, S, prob_flat
        gc.collect() 
        
        annealing_start = torch.tensor(0.01, dtype=torch.float32)
        #annealing_AU = annealing_start * torch.exp(-torch.log(annealing_start) / (total_epoch - good_model_step) * (current_epoch - good_model_step)) if current_epoch >= good_model_step else torch.tensor(0.0).to(device)
        annealing_AU = (annealing_start * torch.exp(-torch.log(annealing_start) / (total_epoch - good_model_step) * (current_epoch - good_model_step))).to(device) if current_epoch >= good_model_step else torch.tensor(0.0).to(device)
        
        
        if current_epoch < good_model_step:
            loss = edl_loss
            noise_loss = torch.tensor(0.0, device=device)
            grad_loss = torch.tensor(0.0, device=device)
        else:

            if gamma != 0 or beta != 0:
                with torch.no_grad():
                    dist_map_np = compute_distance_map(labels).reshape(N, D, H, W)
                    dist_map = torch.from_numpy(dist_map_np).float().to(device)

            if gamma != 0:
                with torch.no_grad():
                    noisy_images = generate_noisy_images_Rician(images, device, mu=0.0, sigma=0.3)  #Rician
                    noisy_u = run_forward_to_get_u_3d(model, noisy_images, C, method='base')
                    dist_map_np = compute_distance_map(labels).reshape(N, D, H, W)
                    dist_map = torch.from_numpy(dist_map_np).float().to(device)
                    
                    # 3D Patch
                    mu1, mu2, d, img_mu1, img_mu2, idxs = sample_class_wise_noised_voxel_patch_images_Rician(
                        images, dist_map, u, labels, k=k, num_patch=num_patch, threshold=d_threshold, device=device)
                    
                out_mu1, _ = model(img_mu1) 
                u_mu1 = get_u_from_model(out_mu1, C).view(N, 1, D, H, W)
                out_mu2, _ = model(img_mu2)
                u_mu2 = get_u_from_model(out_mu2, C).view(N, 1, D, H, W)
                u_ori = u.unsqueeze(1).detach()
                
                noise_loss_mu = L_noise_mu(u_ori, u_mu1, u_mu2, mu1, mu2, d, idxs, threshold=d_threshold, device=device)
                noise_loss_d = L_noise_d(u_ori, u_mu1, u_mu2, d, idxs, threshold=d_threshold, device=device)
                noise_loss_far = L_noise_far(u_ori, u_mu1, u_mu2, dist_map, threshold=d_threshold, device=device)
                
                noise_loss = (noise_loss_mu + noise_loss_d + noise_loss_far) * annealing_AU
                running_noise_loss_mu += noise_loss_mu.item()
                running_noise_loss_d += noise_loss_d.item()
                running_noise_loss_far += noise_loss_far.item()
            else:
                noise_loss = torch.tensor(0.0, device=device)
            if beta != 0:
                # 3D
                grad_map = compute_feature_gradient(encoder_feat)
                grad_loss = L_gradient(u.view(-1, 1), grad_map, dist_map.view(-1, 1), batch_size, device=device) * annealing_AU
                
            else:
                grad_loss = torch.tensor(0.0, device=device)

            loss = edl_loss + (beta * grad_loss) + (gamma * noise_loss)
            #for devis
            #loss = edl_loss


        loss.backward()
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    writer.add_scalar('(Train) Loss/Total', running_loss / len(dataloader), current_epoch)
    
    if 'Heart' in dataset:
        avg_loss = running_loss / len(dataloader)
        avg_dice_LA = running_dice_LA / len(dataloader)
        avg_kl = running_kl_loss / len(dataloader)
        avg_dice_loss = running_dice_loss / len(dataloader)
        avg_ce = running_ce_loss / len(dataloader)
        avg_grad = running_grad_loss / len(dataloader)
        avg_noise = running_noise_loss / len(dataloader)
        
        writer.add_scalar('(Train) Loss/KL', avg_kl, current_epoch)
        writer.add_scalar('(Train) Loss/Dice', avg_dice_loss, current_epoch)
        writer.add_scalar('(Train) Loss/CE', avg_ce, current_epoch)
        writer.add_scalar('(Train) Dice/LA', avg_dice_LA, current_epoch)
        writer.add_scalar('(Train) Loss/Gradient', avg_grad, current_epoch)
        writer.add_scalar('(Train) Loss/Noise_Total', avg_noise, current_epoch)
        
        return avg_loss, avg_dice_LA, avg_kl, avg_dice_loss, avg_ce, avg_grad, avg_noise
        
    elif 'Brain' in dataset:
        avg_wt = running_dice_WT / len(dataloader)
        avg_tc = running_dice_TC / len(dataloader)
        avg_et = running_dice_ET / len(dataloader)
        
        avg_loss = running_loss / len(dataloader)
        avg_kl = running_kl_loss / len(dataloader)
        avg_dice_loss = running_dice_loss / len(dataloader)
        avg_ce = running_ce_loss / len(dataloader)
        avg_grad = running_grad_loss / len(dataloader)
        avg_noise = running_noise_loss / len(dataloader)
        
        writer.add_scalar('(Train) Loss/KL', avg_kl, current_epoch)
        writer.add_scalar('(Train) Loss/Dice', avg_dice_loss, current_epoch)
        writer.add_scalar('(Train) Loss/CE', avg_ce, current_epoch)
        writer.add_scalar('(Train) Dice/WT_Whole', avg_wt, current_epoch)
        writer.add_scalar('(Train) Dice/TC_Core', avg_tc, current_epoch)
        writer.add_scalar('(Train) Dice/ET_Enhancing', avg_et, current_epoch)
        writer.add_scalar('(Train) Loss/Gradient', avg_grad, current_epoch)
        writer.add_scalar('(Train) Loss/Noise_Total', avg_noise, current_epoch)

        return avg_loss, avg_wt, avg_tc, avg_et, avg_kl, avg_dice_loss, avg_ce, avg_grad, avg_noise
       

def get_u_from_model(logits, num_classes):
    N, C, D, H, W = logits.shape
    logits_flat = logits.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
    
    evidence = F.softplus(logits_flat)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    u_flat = num_classes / S # [N*D*H*W, 1]
    u_3d = u_flat.view(N, D, H, W).unsqueeze(1) # [N, 1, D, H, W]
    
    return u_3d




















"""
not used
"""
    
def validate(model,
             dataloader,
             num_classes,
             criterion,
             current_epoch,
             device,
             loss_type,
             writer,
             annealing_steps,
             total_epoch,
             dataset=None,
             **kwargs):
    model.eval()
    running_loss = 0.0
    running_dice_loss = 0.0
    running_kl_loss = 0.0
    
    if 'ACDC' in dataset:
        running_dice_RV = 0.0
        running_dice_Myo = 0.0
        running_dice_LV = 0.0
    elif 'Refuge' in dataset:
        running_dice_DISC = 0.0
        running_dice_CUP = 0.0
    
    C = num_classes
    with torch.no_grad():
        for batch_idx, (images, labels, spacing) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            targets = labels.permute(0, 2, 3, 1).contiguous().view(-1, C)
            
            # N, _, H, W = images.size()
            pred = model(images).permute(0, 2, 3, 1).contiguous().view(-1, C)
            evidence = F.softplus(pred)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            prob = alpha / S
            
            edl_loss, term_ace, kl_loss, dice_loss = criterion(targets, 
                                                               alpha, 
                                                               num_classes=num_classes, 
                                                               current_epoch=current_epoch, 
                                                               total_epoch=total_epoch,
                                                               annealing_steps=annealing_steps, 
                                                               device=device, 
                                                               loss_type=loss_type)
            if 'ACDC' in dataset:
                dice = calculate_dice(targets, prob, epsilon=1e-5, device=device, num_classes=num_classes)
                running_dice_RV += dice[0].item()
                running_dice_Myo += dice[1].item()
                running_dice_LV += dice[2].item()
            elif 'Refuge' in dataset:
                dice_DISC, dice_CUP = calculate_dice_Refuge(targets, prob, epsilon=1e-5, device=device)
                running_dice_DISC += dice_DISC.item()
                running_dice_CUP += dice_CUP.item()
            
            running_loss += edl_loss.item()
            running_dice_loss += dice_loss.item()
            running_kl_loss += kl_loss.item()
            
    writer.add_scalar('(Validate) Loss', running_loss / len(dataloader), current_epoch)
    writer.add_scalar('(Validate) Loss Dice', running_dice_loss / len(dataloader), current_epoch)
    writer.add_scalar('(Validate) Loss KL', running_kl_loss / len(dataloader), current_epoch)
    
    if dataset == 'ACDC':
        writer.add_scalar('(Validate) SDice-RV', running_dice_RV / len(dataloader), current_epoch)
        writer.add_scalar('(Validate) SDice-Myo', running_dice_Myo / len(dataloader), current_epoch)
        writer.add_scalar('(Validate) SDice-LV', running_dice_LV / len(dataloader), current_epoch)
        return running_loss / len(dataloader), running_dice_RV / len(dataloader), running_dice_Myo / len(dataloader), running_dice_LV / len(dataloader)
    elif dataset == 'Refuge':
        writer.add_scalar('(Validate) SDice-DISC', running_dice_DISC / len(dataloader), current_epoch)
        writer.add_scalar('(Validate) SDice-CUP', running_dice_CUP / len(dataloader), current_epoch)
        return running_loss / len(dataloader), running_dice_DISC / len(dataloader), running_dice_CUP / len(dataloader)
    