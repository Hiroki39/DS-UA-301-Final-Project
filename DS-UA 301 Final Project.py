#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import json
import ray
import psutil
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torchvision import utils, transforms
import torch.nn.functional as F
from extras.anchors import get_offsets
from extras.boxes import box_iou, nms
from extras.util import *
from extras.image_manip import ManipDetectionModel
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from ray import tune
from ray.tune.schedulers import HyperBandScheduler

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)


# In[5]:


# to make sure Raytune could work properly it is recommended to use absolute path
COCO_DIR = "/scratch/hz2212/Final Project/train2014"
SYNTHETIC_DIR = "/scratch/hz2212/Final Project/coco_synthetic"
MODEL_DIR = "/scratch/hz2212/Final Project/models"

TRAIN_FILE = "train_filter.txt"
TEST_FILE = "test_filter.txt"


# In[6]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True


# In[7]:


train_file_info_frame = pd.read_csv(TRAIN_FILE, delimiter=" ", header=None)
train_file_info_frame


# In[8]:


test_file_info_frame = pd.read_csv(TEST_FILE, delimiter=" ", header=None)
test_file_info_frame


# In[9]:


def get_image(filenames):
    if isinstance(filenames, str):
        imdir = SYNTHETIC_DIR if filenames[:2] == "Tp" else COCO_DIR
        return io.imread(os.path.join(imdir, filenames))
    else:
        paths = []
        for filename in filenames:
            imdir = SYNTHETIC_DIR if filename[:2] == "Tp" else COCO_DIR
            paths.append(os.path.join(imdir, filename))
        return io.imread_collection(paths)


# In[10]:


class ImageManipDataset(Dataset):

    def __init__(self, txt_file, transform=None, test_mode=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_info_frame = pd.read_csv(txt_file, delimiter=" ", header=None)
        if test_mode:
            self.file_info_frame = pd.read_csv(txt_file, delimiter=" ", header=None).head(2048)
        self.transform = transform
        
        self.images = get_image(self.file_info_frame.iloc[:, 0].values)
        self.bboxs = self.file_info_frame.iloc[:, 1:5].values
        self.is_authentics = (self.file_info_frame.iloc[:, 5] == "authentic").astype(int)
        
        images = []
        
        for idx in range(len(self.file_info_frame)):
            sample = {'image': self.images[idx], 'bbox': self.bboxs[idx].reshape(1, -1)}

            if self.transform:
                sample = self.transform(sample)
            
            images.append(sample['image'])
            self.bboxs[idx] = sample['bbox']
        
        self.images = images
            

#         self.sample = []
        
#         for idx in range(len(self.file_info_frame)):
#             img = get_image(self.file_info_frame.iloc[idx, 0])
#             bbox = self.file_info_frame.iloc[idx, 1:5].values
#             is_authentic = 1 if self.file_info_frame.iloc[idx, 5] == "authentic" else 0
            
#             sample = {'image': img, 'bbox': bbox.reshape(1, -1)}

#             if self.transform:
#                 sample = self.transform(sample)

#             sample["authentic"] = is_authentic
        
#             self.sample.append(sample)
            
    def __len__(self):
        return len(self.file_info_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            'image': self.images[idx],
            'bbox': self.bboxs[idx],
            'authentic': self.is_authentics[idx]
        }

# class ImageManipDataset(Dataset):

#     def __init__(self, txt_file, transform=None, test_mode=False):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.file_info_frame = pd.read_csv(txt_file, delimiter=" ", header=None)
#         if test_mode:
#             self.file_info_frame = pd.read_csv(txt_file, delimiter=" ", header=None).head(2048)
#         self.transform = transform

    
#     def __len__(self):
#         return len(self.file_info_frame)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         image = get_image(self.file_info_frame.iloc[idx, 0])
#         bbox = self.file_info_frame.iloc[idx, 1:5].values
#         is_authentic = 1 if self.file_info_frame.iloc[idx, 5] == "authentic" else 0
#         sample = {'image': image, 'bbox': bbox.reshape(1, -1)}

#         if self.transform:
#             sample = self.transform(sample)
        
#         sample["authentic"] = is_authentic

#         return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bboxs = sample['image'], sample['bbox']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        bboxs = (bboxs * [new_w / w, new_h / h, new_w / w, new_h / h]).astype(float)

        return {'image': img, 'bbox': bboxs}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'bbox': torch.from_numpy(bbox)}


# In[11]:


coco_transform = transforms.Compose([
    Rescale((128, 128)),
    ToTensor()
])


# In[ ]:


transformed_train = ImageManipDataset(txt_file=TRAIN_FILE,
                                      transform=coco_transform, test_mode=False)
transformed_test = ImageManipDataset(txt_file=TEST_FILE,
                                     transform=coco_transform, test_mode=False)


# In[ ]:


# train_loader = DataLoader(transformed_train, batch_size=256, shuffle=True, pin_memory=True, num_workers=8)
# test_loader = DataLoader(transformed_test, batch_size=256, shuffle=False, pin_memory=True, num_workers=8)


# In[ ]:


def get_gt_boxes():
    """
    Generate 192 boxes where each box is represented by :
    [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

    Each anchor position should generate 3 boxes according to the scales and ratios given.

    Return this result as a numpy array of size [192,4]
    """
    stride = 16 # The stride of the final feature map is 16 (the model compresses the image from 128 x 128 to 8 x 8)
    map_sz = 128 # this is the length of height/width of the image

    scales = torch.tensor([10,20,30,40,50,60,70,80,90,100])
    ratios = torch.tensor([[1,1], [0.7, 1.4], [1.4, 0.7], [0.8, 1.2], [1.2, 0.8], [0.6, 1.8], [1.8, 0.6]]).view(1, 14)
    
    half_stride = int(stride / 2)
    num_grids = int((map_sz / stride) ** 2)
    boxes_size = (ratios.T * scales).T.reshape(-1, 2)
    num_boxes = boxes_size.shape[0] * num_grids
    gt_boxes = torch.zeros((num_boxes, 4))

    for i in range(num_boxes):
        grid_index = i // (scales.shape[0] * ratios.shape[1] // 2)
        box_index = i % (scales.shape[0] * ratios.shape[1] // 2)
        center_x = int(grid_index % (map_sz / stride) * stride + half_stride)
        center_y = int(grid_index // (map_sz / stride) * stride + half_stride)
        top_left_x = center_x - (boxes_size[box_index, 0] / 2)
        top_left_y = center_y - (boxes_size[box_index, 1] / 2)
        bottom_right_x = center_x + (boxes_size[box_index, 0] / 2)
        bottom_right_y = center_y + (boxes_size[box_index, 1] / 2)
        gt_boxes[i, :] = torch.tensor([top_left_x, top_left_y, 
                                       bottom_right_x, bottom_right_y])


    return gt_boxes


# In[ ]:


gt_boxes = get_gt_boxes().to(device)


# In[ ]:


def get_bbox_gt(ex_boxes, gt_boxes, is_auth, image_size=128):
    '''

    INPUT:
    ex_boxes: [Nx4]: Bounding boxes in the image. Here N is the number of bounding boxes the image has
    gt_boxes: [192 x 4]: Anchor boxes of an image of size 128 x 128 with stride 16.
    sz : 128
    OUTPUT:
    gt_classes: [192 x 1] : Class labels for each anchor: 1 is for foreground, 0 is for background and -1 is for a bad anchor. [where IOU is between 0.3 and 0.7]
    gt_offsets: [192 x 4]: Offsets for anchor to best fit the bounding box object. 0 values for 0 and -1 class anchors.

    '''
    high_threshold = 0.7
    low_threshold = 0.3

    iou, ex_index = box_iou(gt_boxes, ex_boxes).max(
        dim=1)  # max iou and the index of bounding box
    # nearest bounding box to each anchor box
    crsp_ex_boxes = ex_boxes[ex_index, :]

    gt_classes = -1 * torch.ones(gt_boxes.shape[0], 1).long().to(device)
    gt_classes[iou > high_threshold] = 2 if is_auth else 1
    gt_classes[iou < low_threshold] = 0

    gt_offsets = get_offsets(gt_boxes, crsp_ex_boxes)
    no_object = ((gt_classes == 0) | (gt_classes == -1)
                 ).nonzero(as_tuple=True)[0]
    gt_offsets[no_object, :] = torch.zeros(1, 4).to(device)
    return gt_classes, gt_offsets


def get_targets(sample, target, is_auth):
    '''
    Input
    target => Set of bounding boxes for each image.
    Sample => Each image
    Output:
    Bounding box offsets and class labels for each anchor.
    '''

    batched_preds = []
    batched_offsets = []
    final_cls_targets = []
    final_box_offsets = []
    for s, t, a in zip(sample, target, is_auth):
        bboxes = t.float().reshape(1, -1).to(device)
        class_targets, box_offsets = get_bbox_gt(bboxes, gt_boxes, a, image_size=128)
        final_cls_targets.append(class_targets)
        final_box_offsets.append(box_offsets)

    final_cls_targets = torch.stack(final_cls_targets, dim=0)
    final_box_offsets = torch.stack(final_box_offsets, dim=0)

    return final_cls_targets, final_box_offsets


# In[ ]:


def class_loss(out_pred, class_targets):

    class_targets_copy = class_targets.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, size_average=True).to(device)
    class_targets_copy = class_targets_copy.squeeze()
    
    # downsample negative samples (pick 20 from 192)
    keep_idx = torch.cartesian_prod(torch.arange(class_targets.shape[0]), torch.randperm(class_targets.shape[1])[:20])
    keep_idx = torch.cat((keep_idx.to(device), torch.argwhere(class_targets_copy > 0)))

    modified_class_targets = (torch.ones_like(class_targets) * -1)
    modified_class_targets[keep_idx[:, 0], keep_idx[:, 1]] = class_targets_copy[keep_idx[:, 0], keep_idx[:, 1]] 
    return criterion(out_pred, modified_class_targets.to(device))


def bbox_loss(out_bbox, box_targets, class_targets):
    # return bounding box offset loss
    box_targets = box_targets.to(device)
    class_targets_copy = class_targets.to(device)
    criterion = nn.SmoothL1Loss().to(device)
    class_targets_copy = class_targets_copy.repeat(1, 1, 4)

    out_bbox[class_targets_copy < 1] = 0
    box_targets[class_targets_copy < 1] = 0
    return criterion(out_bbox, box_targets)


# In[ ]:


# Training Function.
def train_manip(config):
    
    epochs=40
    
    model = ManipDetectionModel(base=config['base'], pretrained=config['pretrained']).to(device)
    model.encoder.to(device)
    
    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    train_loader = DataLoader(transformed_train, batch_size=256, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(transformed_test, batch_size=256, shuffle=False, pin_memory=True, num_workers=4)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    model_filename = "model_resnet" + str(config['base']) + "_lr" + str(config['lr'])
    if config['pretrained']:
        model_filename += '_pretrained'
    
    avg_b_train_losses = []
    avg_c_train_losses = []
    avg_b_test_losses = []
    avg_c_test_losses = []
    
    for i in range(epochs):
        
        total_train_loss = 0
        b_train_loss = 0
        c_train_loss = 0

        model.train()
        
        with tqdm(train_loader, desc='Train Progress', unit="batch") as tepoch:
            for data_dict in tepoch:
                ims = data_dict["image"].float().to(device)
                class_targets, box_targets = get_targets(data_dict["image"].to(device), data_dict["bbox"].to(device), data_dict["authentic"].to(device))
                out_pred, out_box = model(ims)

                loss_cls = class_loss(out_pred, class_targets.squeeze(2))
                loss_bbox = bbox_loss(out_box, box_targets, class_targets)

                loss = loss_cls + loss_bbox

                if loss.item() != 0:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                total_train_loss += loss.item()
                c_train_loss += loss_cls.item()
                b_train_loss += loss_bbox.item()

        avg_c_train_loss = float(c_train_loss / len(train_loader))
        avg_b_train_loss = float(b_train_loss / len(train_loader))
        
        avg_c_train_losses.append(avg_c_train_loss)
        avg_b_train_losses.append(avg_b_train_loss)
        
        total_test_loss = 0
        b_test_loss = 0
        c_test_loss = 0
        
        with torch.no_grad():
            
            model.eval()

            with tqdm(test_loader, desc='Test Progress', unit="batch") as tepoch:
                
                for data_dict in tepoch:
                    ims = data_dict["image"].float().to(device)
                    class_targets, box_targets = get_targets(data_dict["image"].to(device), data_dict["bbox"].to(device), data_dict["authentic"].to(device))
                    out_pred, out_box = model(ims)

                    loss_cls = class_loss(out_pred, class_targets.squeeze(2))
                    loss_bbox = bbox_loss(out_box, box_targets, class_targets)

                    loss = loss_cls + loss_bbox

                    total_test_loss += loss.item()
                    c_test_loss += loss_cls.item()
                    b_test_loss += loss_bbox.item()
                    
                avg_c_test_loss = float(c_test_loss / len(test_loader))
                avg_b_test_loss = float(b_test_loss / len(test_loader))

                avg_b_test_losses.append(avg_b_test_loss)
                avg_c_test_losses.append(avg_c_test_loss)

        scheduler.step()

        print('Trained Epoch: {} | Avg Classification Train Loss: {}, Bounding Train Loss: {}, Classification Test Loss: {}, Bounding Test Loss: {}\n'.format(
            i, avg_c_train_loss, avg_b_train_loss, avg_c_test_loss, avg_b_test_loss))

    if device == 'cuda' and torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), os.path.join(MODEL_DIR, model_filename))
    else:
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_filename))
    
    return avg_b_train_losses, avg_c_train_losses, avg_b_test_losses, avg_c_test_losses
    


# In[ ]:


loss_dict = {}

for base in [18, 34, 50]:
    for lr in [0.01, 0.1, 1.0]:
        for pretrained in [True, False]:
            config = {"base": base, "lr": lr, "pretrained": pretrained}
            reg_train_losses, clf_train_losses, reg_test_losses, clf_test_losses = train_manip(config)
            loss_dict[str(base) + '_' + str(lr) + '_' + str(pretrained)] = {
                'reg_train_losses': reg_train_losses,
                'clf_train_losses': clf_train_losses,
                'reg_test_losses': reg_test_losses,
                'clf_test_losses': clf_test_losses
            }


# In[2]:


with open("performance.json", "w") as outfile:
    json.dump(loss_dict, outfile)

for key in loss_dict:
    print(key + " Regression Loss: " + loss_dict[key][0][-1] + " Classification Loss " + loss_dict[key][1][-1])


# In[ ]:


# # fix the issue on colab/hpc
# ray._private.utils.get_system_memory = lambda: psutil.virtual_memory().total


# In[ ]:


# hyperband_scheduler = HyperBandScheduler(
#     time_attr='training_iteration',
#     metric='weighted_loss',
#     mode='min',
#     max_t=10,
#     reduction_factor=3
# )

# analysis = tune.run(
#     train_manip,
#     resources_per_trial={
#         "gpu": 0.33,
#         "cpu": 8
#     },
#     config={
#         "base": tune.grid_search([18, 34, 50]),
#         "lr": tune.grid_search([0.01, 0.1, 1.0]),
#         "pretrained": tune.grid_search([True, False])
#     },
#     verbose=1,
#     scheduler=hyperband_scheduler
# )

