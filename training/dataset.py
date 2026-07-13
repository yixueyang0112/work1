import os
import glob
import torch
import random
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset

from PIL import Image, ImageOps
from preprocessing.resampling import resample_data_or_seg
from utilities.utils import RandomCenterCrop, RandomFlip, RandomRotate, RandomScaleCenterCrop, \
    eraser, elastic_transform_3d, add_salt_pepper_noise, adjust_light, Normalize_tf, ToTensor, Scale
from torchvision import transforms



def obtain_filenames_from_id_list(datapath, id_list=None):
    Files = []
    search_dir = os.path.join(datapath, "imagesTr")
    if id_list is not None:
        for patient_id in id_list:
            search_pattern = os.path.join(search_dir, f'{patient_id}_mri.nii.gz')
            matched_files = glob.glob(search_pattern)
            if len(matched_files) == 0:
                print(f"Warning: No file found for ID {patient_id:03d} in {search_dir}")
            Files.extend(matched_files)
    else:
        Files.extend(glob.glob(os.path.join(search_dir, '*.nii.gz')))
    Files.sort()
    return Files


def obtain_brain_filenames(all_img_paths, id_list):
    return [all_img_paths[i] for i in id_list]


def Generate_Heart_Train_Val_Test_List(datapath, seed=1):
    np.random.seed(seed)
    
    search_path = os.path.join(datapath, "imagesTr", "*_mri.nii.gz")
    all_img_paths = sorted(glob.glob(search_path))
    
    patient_ids = []
    for path in all_img_paths:
        name = os.path.basename(path)
        p_id = name.split('_')[0] 
        patient_ids.append(p_id)

    patient_ids = np.array(patient_ids)
    num_total = len(patient_ids)

    shuffled_indices = np.random.permutation(np.arange(num_total))
    shuffled_ids = patient_ids[shuffled_indices]

    # 0.8train 0.1vali 0.1test
    num_train = int(num_total * 0.8)
    num_val = int(num_total * 0.1)
    num_test = num_total - num_train - num_val 

    train_ids = shuffled_ids[0 : num_train].tolist()
    vali_ids = shuffled_ids[num_train : num_train + num_val].tolist()
    test_ids = shuffled_ids[num_train + num_val : ].tolist()

    train_filenames = obtain_filenames_from_id_list(datapath, train_ids)
    vali_filenames = obtain_filenames_from_id_list(datapath, vali_ids)
    test_filenames = obtain_filenames_from_id_list(datapath, test_ids)

    Train_Files = [train_filenames]
    Vali_Files = [vali_filenames]
    Test_Files = test_filenames
    return Train_Files, Vali_Files, Test_Files



def Generate_Brain_Train_Val_Test_List(datapath, seed=1):
    np.random.seed(seed)
    
    all_patient_dirs = sorted([
        os.path.join(datapath, d) for d in os.listdir(datapath) 
        if os.path.isdir(os.path.join(datapath, d)) and not d.startswith('.')
    ])
    
    num_total = len(all_patient_dirs)
    shuffled_indices = np.random.permutation(np.arange(num_total))
    shuffled_patient_dirs = [all_patient_dirs[i] for i in shuffled_indices]

    # 0.8train 0.1vali 0.1test
    num_train = int(num_total * 0.8)
    num_val = int(num_total * 0.1)
    num_test = num_total - num_train - num_val 

    train_filenames = shuffled_patient_dirs[0 : num_train]
    vali_filenames = shuffled_patient_dirs[num_train : num_train + num_val]
    test_filenames = shuffled_patient_dirs[num_train + num_val : ]

    Train_Files = [train_filenames]
    Vali_Files = [vali_filenames]
    Test_Files = test_filenames
    return Train_Files, Vali_Files, Test_Files


class Heart_Dataset(Dataset):
    def __init__(self, ImgFiles, times, num_classes=2, train=True, transform=None, patch_size=(128,128,128)):
     
        self.ImgFiles = ImgFiles
        self.time = times
        self.num_classes = num_classes
        self.train = train
        self.transform = transform
        self.patch_size = patch_size
        
        sitk.ProcessObject_SetGlobalWarningDisplay(False)

    def __len__(self):
        return len(self.ImgFiles) * self.time
   
    def __getitem__(self, item):
        
        item, _ = divmod(item, self.time)
        img_path = self.ImgFiles[item]
        lab_path = img_path.replace('imagesTr', 'labelsTr').replace('mri', 'label')
        img_name = os.path.basename(img_path)

        # Data load
        itkimg = sitk.ReadImage(img_path)
        itklab = sitk.ReadImage(lab_path)
        np_img = sitk.GetArrayFromImage(itkimg).astype(np.float32)
        np_lab = sitk.GetArrayFromImage(itklab).astype(np.uint8)
        spacing = np.array(itkimg.GetSpacing()).astype(np.float32)
        np_lab[np_lab == 255] = 1 

        d, h, w = np_img.shape
        td, th, tw = self.patch_size
        
        # padding
        pad_d = max(0, td - d)
        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            np_img = np.pad(np_img, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            np_lab = np.pad(np_lab, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            d, h, w = np_img.shape
        
        if self.train:
            if np.any(np_lab > 0):
                z_indices, y_indices, x_indices = np.where(np_lab > 0)
                z_center = int(np.mean(z_indices))
                y_center = int(np.mean(y_indices))
                x_center = int(np.mean(x_indices))

                ctrl_d = z_center - td // 2 + np.random.randint(-15, 16)
                ctrl_h = y_center - th // 2 + np.random.randint(-15, 16)
                ctrl_w = x_center - tw // 2 + np.random.randint(-15, 16)
                
                ctrl_d = max(0, min(ctrl_d, d - td))
                ctrl_h = max(0, min(ctrl_h, h - th))
                ctrl_w = max(0, min(ctrl_w, w - tw))
            else:
                ctrl_d = np.random.randint(0, d - td + 1)
                ctrl_h = np.random.randint(0, h - th + 1)
                ctrl_w = np.random.randint(0, w - tw + 1)
        else:
            if np.any(np_lab > 0):
                z_indices, y_indices, x_indices = np.where(np_lab > 0)
                ctrl_d = max(0, min(int(np.mean(z_indices)) - td // 2, d - td))
                ctrl_h = max(0, min(int(np.mean(y_indices)) - th // 2, h - th))
                ctrl_w = max(0, min(int(np.mean(x_indices)) - tw // 2, w - tw))
            else:
                ctrl_d = (d - td) // 2
                ctrl_h = (h - th) // 2
                ctrl_w = (w - tw) // 2


        np_img = np_img[ctrl_d:ctrl_d+td, ctrl_h:ctrl_h+th, ctrl_w:ctrl_w+tw]
        np_lab = np_lab[ctrl_d:ctrl_d+td, ctrl_h:ctrl_h+th, ctrl_w:ctrl_w+tw]

        np_img_norm = (np_img - np_img.mean()) / (np_img.std() + 1e-8)
        img_tensor = torch.from_numpy(np_img_norm).float().unsqueeze(0)
        
        # one-hot
        mask_idx = np_lab.astype(np.int64)
        one_hot_lab = np.eye(self.num_classes)[mask_idx]         # [D, H, W, num_classes]
        one_hot_lab = np.transpose(one_hot_lab, (3, 0, 1, 2))    # [num_classes, D, H, W]
        lab_tensor = torch.from_numpy(one_hot_lab).long()

        #print(f"Image shape: {img_tensor.shape}, Label shape: {lab_tensor.shape}")
        return img_tensor, lab_tensor, spacing



class Brain_Dataset(Dataset):

    def __init__(self, ImgFiles, times, num_classes=4, train=True, transform=None, patch_size=(128, 128, 128)):
        self.ImgFiles = ImgFiles
        self.time = times
        self.num_classes = num_classes
        self.train = train
        self.transform = transform
        self.patch_size = patch_size
        
        sitk.ProcessObject_SetGlobalWarningDisplay(False)

    def __len__(self):
        return len(self.ImgFiles) * self.time

    def __getitem__(self, item):
        item, _ = divmod(item, self.time)
        patient_dir = self.ImgFiles[item]
        img_name = os.path.basename(patient_dir)

        flair_path = os.path.join(patient_dir, f"{img_name}_flair.nii")
        t1_path = os.path.join(patient_dir, f"{img_name}_t1.nii")
        t1ce_path = os.path.join(patient_dir, f"{img_name}_t1ce.nii")
        t2_path = os.path.join(patient_dir, f"{img_name}_t2.nii")
        lab_path = os.path.join(patient_dir, f"{img_name}_seg.nii")

        np_flair = sitk.GetArrayFromImage(sitk.ReadImage(flair_path)).astype(np.float32)
        np_t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1_path)).astype(np.float32)
        np_t1ce = sitk.GetArrayFromImage(sitk.ReadImage(t1ce_path)).astype(np.float32)
        np_t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2_path)).astype(np.float32)
        
        np_img = np.stack([np_flair, np_t1, np_t1ce, np_t2], axis=0)
        
        itklab = sitk.ReadImage(lab_path)
        np_lab = sitk.GetArrayFromImage(itklab).astype(np.uint8)

        target_lab = np.zeros_like(np_lab)
        target_lab[np_lab == 1] = 1 # NCR/NET
        target_lab[np_lab == 2] = 2 # ED
        target_lab[np_lab == 4] = 3 # ET
        
        spacing = np.array(itklab.GetSpacing()).astype(np.float32)


        
        mask_idx = target_lab.astype(np.int64)
        one_hot_lab = np.eye(self.num_classes)[mask_idx]
        one_hot_lab = np.transpose(one_hot_lab, (3, 0, 1, 2))

        # cropping
        z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(np_img, axis=0) != 0)
        
        zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
        zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
        
        # delete background
        np_img = np_img[:, zmin:zmax, ymin:ymax, xmin:xmax]
        one_hot_lab = one_hot_lab[:, zmin:zmax, ymin:ymax, xmin:xmax]
        
        if self.train:
            np_img_crop, np_lab_crop = pad_or_crop_image(np_img, one_hot_lab, target_size=self.patch_size)
        else:
            np_img_crop, np_lab_crop = np_img, one_hot_lab
        
        for c in range(np_img_crop.shape[0]):
            m, s = np_img_crop[c].mean(), np_img_crop[c].std()
            np_img_crop[c] = (np_img_crop[c] - m) / (s + 1e-8)
            
        img_tensor = torch.from_numpy(np_img_crop).float() 
        lab_tensor = torch.from_numpy(np_lab_crop).float()

        #print(f"Image shape: {img_tensor.shape}, Label shape: {lab_tensor.shape}")
        return img_tensor, lab_tensor, spacing


def pad_or_crop_image(image, seg=None, target_size=(128, 144, 144)):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    image = image[:, z_slice, y_slice, x_slice]
    if seg is not None:
        seg = seg[:, z_slice, y_slice, x_slice]
    todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    padlist = [(0, 0)]  # channel dim
    for to_pad in todos:
        if to_pad[0]:
            padlist.append((to_pad[1], to_pad[2]))
        else:
            padlist.append((0, 0))
    image = np.pad(image, padlist)
    if seg is not None:
        seg = np.pad(seg, padlist)
        return image, seg
    return image



def get_crop_slice(target_size, dim_size):
    if dim_size > target_size:
        start = (dim_size - target_size) // 2
        end = start + target_size
        return slice(start, end)
    else:
        return slice(0, dim_size)


def get_left_right_idx_should_pad(target_size, dim_size):
    if dim_size < target_size:
        diff = target_size - dim_size
        left = diff // 2
        right = diff - left
        return True, left, right
    else:
        return False, 0, 0
