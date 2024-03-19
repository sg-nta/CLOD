#------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse
import json
import random
import time
import pickle
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
import torch.distributed as dist
from Custom_Dataset import *
from custom_utils import *
from custom_prints import *
from custom_buffer_manager import *
from custom_training import rehearsal_training

from datasets import build_dataset, get_coco_api_from_dataset
from main_component import TrainingPipeline, generate_dataset
from glob import glob
import wandb

from configs.arguments import get_args_parser, deform_detr_parser


def main(args):
    #* Initializing
    pipeline = TrainingPipeline(args)
    args = pipeline.args

    #* Evaluation mode
    if args.eval:
        pipeline.evaluation_only_mode()
        return
    
    #* Pseudo labeling mode in generated dataset
    if args.pseudo_labeling:
        pipeline.pseudo_work()
        return
    
    #* Pseudo labeling mode in generated dataset
    if args.labeling_check:
        pipeline.labeling_check()
        return
    
    #* No incremental learning process, only normal training
    if pipeline.tasks == 1 :
        pipeline.set_task_epoch(args, 0) # only first task
        pipeline.only_one_task_training()
        return
    
    #* CL training
    print("Start training")
    start_time = time.time()
    is_task_changed = False
    
    pipeline.load_ddp_state()
    for idx, task_idx in enumerate(range(pipeline.start_task, pipeline.tasks)):
        last_task = (task_idx+1 == pipeline.tasks)
        first_training = (task_idx == 0)

        if is_task_changed and args.Branch_Incremental:
            pipeline.make_branch(task_idx, args, is_init=False)
            is_task_changed = False
        
        dataset_train, data_loader_train, sampler_train, list_CC = generate_dataset(first_training, task_idx, args, pipeline)
        
        # Incremental training for each epoch
        pipeline.set_task_epoch(args, idx)
        pipeline.incremental_train_epoch(task_idx=task_idx, last_task=last_task, dataset_train=dataset_train,
                                        data_loader_train=data_loader_train, sampler_train=sampler_train,
                                        list_CC=list_CC, first_training=first_training)
        
        is_task_changed = True
        
    # Calculate and print the total time taken for training
    import datetime
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training completed in: ", total_time_str)

    
import warnings
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning) 
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    parent_args = parser.parse_known_args()[0]

    # set parser
    if parent_args.model_name == 'deform_detr':
        parser = deform_detr_parser(parser)
        args = parser.parse_args()
    else:
        msg = 'Unsupported model name!'
        raise Exception(msg)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
