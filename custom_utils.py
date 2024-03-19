from ast import arg
from pyexpat import model
from sched import scheduler
from xmlrpc.client import Boolean
import torch
import util.misc as utils
import numpy as np
from typing import Tuple, Dict, List, Optional
import os
from custom_prints import over_label_checker, check_components
from termcolor import colored
from torch.utils.data import DataLoader
from datasets import build_dataset, get_coco_api_from_dataset


def new_dataLoader(saved_dict, args):
    dataset_idx_list = []
    for _, value in saved_dict.items():
        if len(value) > 0 :
            np_idx_list = np.array(value, dtype=object)
            dataset_idx_list.extend(np.unique(np_idx_list[:, 3]).astype(np.uint8).tolist())
    
    custom_dataset = build_dataset(image_set='train', args=args, img_ids=dataset_idx_list)
    
    custom_loader = DataLoader(custom_dataset, args.batch_size, shuffle=True, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    return custom_loader


def load_model_params(mode, model: model, dir: str = None, Branch_Incremental = False):
    new_model_dict = model.state_dict()
    
    if isinstance(dir, list):
        dir = dir[0]
    #temp dir
    checkpoint = torch.load(dir)
    pretraind_model = checkpoint["model"]
    name_list = [name for name in new_model_dict.keys() if name in pretraind_model.keys()]

    if mode != 'eval' and Branch_Incremental:
        name_list = list(filter(lambda x : "class" not in x, name_list))
        name_list = list(filter(lambda x : "label" not in x, name_list)) # for dn_detr
    pretraind_model_dict = {k : v for k, v in pretraind_model.items() if k in name_list } # if "class" not in k => this method used in diff class list
    
    new_model_dict.update(pretraind_model_dict)
    model.load_state_dict(new_model_dict)
    print(colored(f"pretrained Model loading complete: {dir}", "blue", "on_yellow"))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    #No parameter update    
    for name, params in model.named_parameters():
        if name in pretraind_model_dict.keys():
            if mode == "teacher":
                params.requires_grad = False #if you wanna set frozen the pre parameters for specific Neuron update, so then you could set False
        else:
            if mode == "teacher":
                params.requires_grad = False
    
    print(colored(f"Done every model params", "red", "on_yellow"))
            
    return model

def teacher_model_freeze(model):
    for _, params in model.named_parameters():
            params.requires_grad = False
                
    return model

def save_model_params(model_without_ddp, optimizer, lr_scheduler, args, output_dir, task_index, total_tasks, epoch=-1):
    """Save model parameters for each task."""
    
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    
    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Determine the checkpoint file name based on task and epoch
    checkpoint_filename = f'cp_{total_tasks:02}_{task_index + 1:02}'
    if epoch != -1:
        checkpoint_filename += f'_{epoch}'
    checkpoint_filename += '.pth'
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    # Save model and other states
    utils.save_on_master({
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'args': args,
    }, checkpoint_path)


import torch.distributed as dist
def check_training_gpu(train_check):
    world_size = utils.get_world_size()
    
    
    if world_size < 2:
        return True
    
    gpu_control_value = torch.tensor(1.0, device=torch.device("cuda"))
    temp_list = [torch.tensor(0.0, device=torch.device("cuda")) for _ in range(4)]
    
    if train_check == False:
        gpu_control_value = torch.tensor(0.0, device=torch.device("cuda"))
        
    dist.all_gather(temp_list, gpu_control_value)
    gpu_control_value = sum([ten_idx.item() for ten_idx in temp_list])
    print(f"used gpu counts : {int(gpu_control_value)}")
    if int(gpu_control_value) == 0:
        print("current using GPU counts is 0, so it's not traing")
        return False

    return True

def buffer_checker(args, task, rehearsal):
    #print text file
    check_components(args, task, rehearsal, True)
        
        
def control_lr_backbone(args, optimizer, frozen):
    if frozen is True:
        lr = 0.0
    else:
        lr = args.lr_backbone
        
    optimizer.param_groups[-1]['lr'] = lr
            
    return optimizer


def dataset_configuration(args, original_set, aug_set):
    
    original_dataset, original_loader, original_sampler = original_set[0], original_set[1], original_set[2]
    if aug_set is None :
        AugRplay_dataset, AugRplay_loader, AugRplay_sampler = None, None, None
    else :
        AugRplay_dataset, AugRplay_loader, AugRplay_sampler = aug_set[0], aug_set[1], aug_set[2]
    
    if args.AugReplay and not args.MixReplay:
        return AugRplay_dataset, AugRplay_loader, AugRplay_sampler
    
    elif args.AugReplay and args.MixReplay:
        print(colored("MixReplay dataset generating", "blue", "on_yellow"))
        return [AugRplay_dataset, original_dataset], [AugRplay_loader, original_loader], [AugRplay_sampler, original_sampler] 
    
    else :
        print(colored("Original dataset generating", "blue", "on_yellow"))
        return original_dataset, original_loader, original_sampler

def compute_location_loss(teacher_outputs):
    """Compute location loss between teacher and student models."""
    t_aux_pred = []

    t_logits, t_boxes = teacher_outputs['pred_logits'], teacher_outputs['pred_boxes']
    t_aux_pred = teacher_outputs["aux_outputs"]
    return t_logits, t_boxes, t_aux_pred

def compute_gen_idx_loss(logits, boxes, aux_outputs, gen_idx):
    gen_logits = logits[gen_idx]
    gen_boxes = boxes[gen_idx]
    gen_aux_logits = [ap["pred_logits"][gen_idx] for ap in aux_outputs]
    gen_aux_boxes = [ap["pred_boxes"][gen_idx] for ap in aux_outputs]
    return gen_logits, gen_boxes, gen_aux_logits, gen_aux_boxes

def compute_real_idx_loss(logits, boxes, aux_outputs, gen_idx):
    real_idx = [i for i in range(len(logits)) if i not in gen_idx]
    real_logits = logits[real_idx]
    real_boxes = boxes[real_idx]
    real_aux_logits = [ap["pred_logits"][real_idx] for ap in aux_outputs]
    real_aux_boxes = [ap["pred_boxes"][real_idx] for ap in aux_outputs]
    return real_logits, real_boxes, real_aux_logits, real_aux_boxes
    
import gc
def refresh_data():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    
    
import os
import shutil
import json
from tqdm import tqdm

def check_and_copy_different_annotations(pseudo, origin, gen_path):
    gen_image_path = os.path.join(gen_path, "images")
    
    # Create specific folder
    destination_folder = os.path.join(gen_path, "diff_anns")
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    pseudo_images = pseudo.getImgIds()
    
    # List to save image IDs with differences and their count
    origin_more = []  # When the number of annotations in origin is greater
    
    for p_img_id in tqdm(pseudo_images, desc="label checking processing "):
        original_annotations_count = len(origin.loadAnns(origin.getAnnIds(p_img_id)))
        pseudo_annotations_count = len(pseudo.loadAnns(pseudo.getAnnIds(p_img_id)))
    
        origin_anns = origin.loadAnns(origin.getAnnIds(p_img_id))
        pseudo_anns = pseudo.loadAnns(pseudo.getAnnIds(p_img_id))
        
        origin_size = (int(origin.loadImgs(p_img_id)[0]["height"]), int(origin.loadImgs(p_img_id)[0]["width"]))
        pseudo_size = (int(pseudo.loadImgs(p_img_id)[0]["height"]), int(pseudo.loadImgs(p_img_id)[0]["width"]))
        
        if original_annotations_count > pseudo_annotations_count:
            original_img_path = os.path.join(gen_image_path, pseudo.loadImgs(p_img_id)[0]["file_name"])
            
            # Draw bounding box on the image
            img_with_bbox = draw_bbox_on_image(original_img_path, origin_anns, pseudo_anns, origin_size, pseudo_size)  # origin in red
            
            # Save the image with the bounding box
            bbox_img_path = os.path.join(destination_folder, pseudo.loadImgs(p_img_id)[0]["file_name"])
            cv2.imwrite(bbox_img_path, img_with_bbox)
            
            # Store only the image id, category id, and instance id
            origin_ids = [(ann["id"], ann["category_id"]) for ann in origin_anns]
            pseudo_ids = [(ann["id"], ann["category_id"]) for ann in pseudo_anns]
            origin_more.append((p_img_id, original_annotations_count, pseudo_annotations_count, origin_ids, pseudo_ids))
    
    # Sort by image ID in descending order
    origin_more.sort(key=lambda x: x[0])
    
    with open(os.path.join(destination_folder, "origin_more.txt"), "a") as txt_file:
        for diff in origin_more:
            txt_file.write(f"Image ID: {diff[0]}, Original Count: {diff[1]}, Pseudo Count: {diff[2]}, origin labels :  {diff[3]}, pseudo labels: {diff[4]}\n")

import cv2
def draw_bbox_on_image(image_path, origin_anns, pseudo_anns, origin_size, pseudo_size):
    # 이미지를 불러옵니다.
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    # origin의 bounding box를 빨간색으로 그립니다. #* (original scene is Total_coco.json)
    for ann in origin_anns:
        bbox = ann['bbox']
        x, y, w_bbox, h_bbox = [int(coord) for coord in bbox]
        # 정규화
        # x_norm, y_norm, w_norm, h_norm = x/origin_size[1], y/origin_size[0], w_bbox/origin_size[1], h_bbox/origin_size[0]
        # x_norm, y_norm, w_norm, h_norm = x, y/origin_size[0], w_bbox, h_bbox/origin_size[0]
        # 512 크기에 맞게 조정
        # x, y, w_bbox, h_bbox = int(x_norm*512), int(y_norm*512), int(w_norm*512), int(h_norm*512)
        cv2.rectangle(img, (x, y), (x+w_bbox, y+h_bbox), (0, 0, 255), 2)
    
    # pseudo의 bounding box를 파란색으로 그립니다. #* pseudo_data.json
    for ann in pseudo_anns:
        bbox = ann['bbox']
        x, y, w_bbox, h_bbox = [int(coord) for coord in bbox]
        # 정규화
        # x_norm, y_norm, w_norm, h_norm = x/pseudo_size[1], y/pseudo_size[0], w_bbox/pseudo_size[1], h_bbox/pseudo_size[0]
        # # 512 크기에 맞게 조정
        # x, y, w_bbox, h_bbox = int(x_norm*512), int(y_norm*512), int(w_norm*512), int(h_norm*512)
        cv2.rectangle(img, (x, y), (x+w_bbox, y+h_bbox), (255, 0, 0), 2)
    
    return img