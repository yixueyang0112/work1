import torch
import torch.nn.functional as F
import random
import numpy as np
from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance
from scipy.ndimage import distance_transform_edt as distance

from model.UNet3D import Uentropy3D


def compute_distance_map(labels):
    labels_np = labels.cpu().detach().numpy()
    
    # if 3d label (N, C, D, H, W)
    if labels_np.ndim == 5:
        N, C, D, H, W = labels_np.shape
        distance2boundary_batch = np.zeros((N, D, H, W))
        
        for n in range(N):
            boundary = np.zeros((D, H, W))
            for c in range(1, C):
                img_gt = labels_np[n, c, :, :, :] # [D, H, W]
                posmask = img_gt > 0
                if posmask.any():
                    boundary += skimage_seg.find_boundaries(posmask, connectivity=3, mode='thick').astype(np.uint16)
            boundary_bool = boundary > 0
            distance2boundary_batch[n] = distance(~boundary_bool) 
        distance2boundary = distance2boundary_batch.reshape(-1, 1)

    # if 2d label (N, C, H, W)
    elif labels_np.ndim == 4:
        N, C, H, W = labels_np.shape
        distance2boundary_batch = np.zeros((N, H, W))
        
        for n in range(N):
            boundary = np.zeros((H, W))
            for c in range(1, C):
                img_gt = labels_np[n, c, :, :]
                posmask = img_gt > 0
                if posmask.any():
                    boundary += skimage_seg.find_boundaries(posmask, connectivity=2, mode='thick').astype(np.uint16)
            
            boundary_bool = boundary > 0
            distance2boundary_batch[n] = distance(~boundary_bool)
            
        distance2boundary = distance2boundary_batch.reshape(-1, 1)
    
    return distance2boundary


def generate_noisy_images_Rician(images, device, mu=0.5, sigma=0.3, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    images = images.to(device)
    shape = images.shape # [N, 1, D, H, W]
    noise_real = torch.normal(mean=0.0, std=sigma, size=shape, device=device)
    noise_imag = torch.normal(mean=0.0, std=sigma, size=shape, device=device)
    # Magnitude = sqrt((Image + mu + noise_real)^2 + (noise_imag)^2)
    noisy_images = torch.sqrt((images + mu + noise_real)**2 + noise_imag**2)
    #return noisy_images
    return torch.clamp(noisy_images, 0.0, 1.0)


def sample_class_wise_noised_voxel_patch_images_Rician(images, distance_map, uncertainty_mask, label, device, k=3, mu_range=(0, 0.2), sigma=0.3, num_patch=4, threshold=4, seed=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    if isinstance(distance_map, torch.Tensor):
        distance_map_tensor = distance_map.to(device).float()
    else:
        distance_map_tensor = torch.from_numpy(distance_map).to(device).float()
    images = images.to(device)
    batch_size, C, D, H, W = images.shape
    num_classes = label.shape[1]

    mu1 = random.uniform(*mu_range)
    mu2 = random.uniform(*mu_range)
    noise_real1 = torch.normal(mean=0.0, std=sigma, size=(C, k, k, k), device=device)
    noise_imag1 = torch.normal(mean=0.0, std=sigma, size=(C, k, k, k), device=device)
    noise_real2 = torch.normal(mean=0.0, std=sigma, size=(C, k, k, k), device=device)
    noise_imag2 = torch.normal(mean=0.0, std=sigma, size=(C, k, k, k), device=device)

    noised_images_mu1 = images.clone().detach()
    noised_images_mu2 = images.clone().detach()

    d_vals = torch.zeros((batch_size, num_patch * num_classes), device=device)
    indexes = []

    for i in range(batch_size):
        indexes_batch = []
        noise_mask = torch.zeros((D, H, W), dtype=torch.bool, device=device)
        starts = []
        
        for c in range(num_classes):
            cls_starts = []
            u_mask_img = uncertainty_mask[i].view(-1)
            label_img = label[i, c].view(-1)
            nonzero_indices = torch.nonzero(torch.logical_and(u_mask_img, label_img))

            if nonzero_indices.numel() == 0:
                possible_indices = torch.tensor([], device=device)
            else:
                possible_indices = nonzero_indices.squeeze(1)

            if possible_indices.numel() != 0:
                possible_indices = possible_indices[torch.randperm(len(possible_indices))]
                for idx in possible_indices:
                    idx_int = int(idx)
                    z = idx_int // (H * W)
                    remain = idx_int % (H * W)
                    y = remain // W
                    x = remain % W
                    z_start, y_start, x_start = z - k // 2, y - k // 2, x - k // 2

                    if (z_start < 0 or y_start < 0 or x_start < 0 or 
                        z_start + k > D or y_start + k > H or x_start + k > W):
                        continue
                    if c == 0 and distance_map_tensor[i, z, y, x] >= threshold:
                        continue
                    if not noise_mask[z_start:z_start + k, y_start:y_start + k, x_start:x_start + k].any():
                        cls_starts.append((z_start, y_start, x_start))
                        starts.append((z_start, y_start, x_start))
                        noise_mask[z_start:z_start + k, y_start:y_start + k, x_start:x_start + k] = True
                    if len(cls_starts) >= num_patch:
                        break

        while len(starts) < num_patch * num_classes:
            z_s, y_s, x_s = random.randint(0, D - k), random.randint(0, H - k), random.randint(0, W - k)
            if not noise_mask[z_s:z_s+k, y_s:y_s+k, x_s:x_s+k].any():
                starts.append((z_s, y_s, x_s))
                noise_mask[z_s:z_s+k, y_s:y_s+k, x_s:x_s+k] = True

        for j, (z_s, y_s, x_s) in enumerate(starts):
            z_c, y_c, x_c = z_s + k // 2, y_s + k // 2, x_s + k // 2
            
            patch_ori = images[i, :, z_s:z_s + k, y_s:y_s + k, x_s:x_s + k]
            res_mu1 = torch.sqrt((patch_ori + mu1 + noise_real1)**2 + noise_imag1**2 + 1e-8)
            noised_images_mu1[i, :, z_s:z_s + k, y_s:y_s + k, x_s:x_s + k] = res_mu1
            
            res_mu2 = torch.sqrt((patch_ori + mu2 + noise_real2)**2 + noise_imag2**2 + 1e-8)
            noised_images_mu2[i, :, z_s:z_s + k, y_s:y_s + k, x_s:x_s + k] = res_mu2
            
            d_vals[i, j] = distance_map_tensor[i, z_c, y_c, x_c]
            indexes_batch.append((z_c, y_c, x_c))
            
        indexes.append(indexes_batch)

    noised_images_mu1 = torch.clamp(noised_images_mu1, 0.0, 1.0)
    noised_images_mu2 = torch.clamp(noised_images_mu2, 0.0, 1.0)

    return mu1, mu2, d_vals, noised_images_mu1, noised_images_mu2, indexes


def run_forward_to_get_u_3d(model, images, num_classes, method='base', dataset='MSD', model_PU=None):
    if method == 'base' or method == 'devis':
        outputs = model(images)
        res = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        
        if method == 'base':
            pred = res.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
            evidence = F.softplus(pred)
        elif method == 'devis':
            evidence = res.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes) 
            
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)  # [N*D*H*W, 1]
        u = (num_classes / S)  # [N*D*H*W, 1]

    elif method in ['pu', 'flow', 'glow', 'udrop']:
        if method in ['pu', 'flow', 'glow']:
            if 'Refuge-no' in dataset:
                resized_images = F.interpolate(images, size=(64, 128, 128), mode='trilinear', align_corners=False)
            else:
                resized_images = images
            
            logits = model_PU(resized_images, model)
            del resized_images
            
            if 'Refuge-no' in dataset:
                logits = F.interpolate(logits, size=images.shape[2:], mode='nearest')
        
        elif method == 'udrop':
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        
        if logits.ndim == 4:
            logits = logits.unsqueeze(0)
            
        u = Uentropy3D(logits, num_classes).view(-1, 1)

    else:
        if method == 'eu':
            for i in range(4):
                outputs = model[i](images)
                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                
                if i == 0:
                    u = Uentropy3D(logits, num_classes).view(-1, 1)
                else:
                    u += Uentropy3D(logits, num_classes).view(-1, 1)
            u /= 4.
            
        elif method == 'tta':
            import ttach as tta
            transforms_img = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.VerticalFlip(), 
                    tta.Rotate90(angles=[0, 180]),
                    tta.Scale(scales=[1, 1.1]),
                ]
            )
            
            tta_model = tta.SegmentationTTAWrapper(model, transforms_img)
            outputs = tta_model(images)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            u = Uentropy3D(logits, num_classes).view(-1, 1)
            
    return u


def compute_feature_gradient(feature_tensor):
    
    if isinstance(feature_tensor, tuple):
        feature_tensor = feature_tensor[0]
    feat_map = torch.mean(feature_tensor, dim=1, keepdim=True) # [N, 1, D, H, W]
    
    
    kernel_z = torch.tensor([[[[-1, 0, 1]]]], device=feature_tensor.device).float().view(1, 1, 3, 1, 1)
    kernel_y = torch.tensor([[[[-1, 0, 1]]]], device=feature_tensor.device).float().view(1, 1, 1, 3, 1)
    kernel_x = torch.tensor([[[[-1, 0, 1]]]], device=feature_tensor.device).float().view(1, 1, 1, 1, 3)
    
    grad_z = F.conv3d(feat_map, kernel_z, padding=(1, 0, 0))
    grad_y = F.conv3d(feat_map, kernel_y, padding=(0, 1, 0))
    grad_x = F.conv3d(feat_map, kernel_x, padding=(0, 0, 1))
    feat_grad = torch.sqrt(grad_z**2 + grad_y**2 + grad_x**2 + 1e-8)
    
    feat_grad = (feat_grad - feat_grad.min()) / (feat_grad.max() - feat_grad.min() + 1e-8)
    return feat_grad.permute(0, 2, 3, 4, 1).contiguous().view(-1, 1)