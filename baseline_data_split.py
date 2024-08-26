import argparse
import logging
import os
import time

import cv2
import torch
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from config.config import config as cfg
from utils.dataset import MXFaceDatasetSplit
from utils.utils_logging import init_logging

torch.backends.cudnn.benchmark = True

def main(args):
    torch.cuda.set_device(0)
    rank = 0 

    if not os.path.exists(cfg.out_teacher_baseline) and rank == 0:
        os.makedirs(cfg.out_teacher_baseline)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.out_teacher_baseline, logfile="data_split.log")

    trainset=MXFaceDatasetSplit(root_dir="/nas-ctm01/datasets/public/BIOMETRICS/race_per_7000_aligned", ethnicity=cfg.ethnicity)
    train_sampler=torch.utils.data.RandomSampler(trainset)
    train_loader=DataLoader(dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=False, prefetch_factor=2)
    
    races=["African", "Asian", "Caucasian","Indian"]
    logging.info("Split starting for %s samples.", cfg.ethnicity)
    for _, (img, split_id, extra_img_path) in enumerate(train_loader):
        img=img.numpy()
        for sample in range(len(split_id)):
            if not os.path.exists("/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/baseline/teacher_"+races[split_id[sample]]+extra_img_path[sample].replace(extra_img_path[sample].split("/")[3],"")):
                os.makedirs("/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/baseline/teacher_"+races[split_id[sample]]+extra_img_path[sample].replace(extra_img_path[sample].split("/")[3],""))
            cv2.imwrite("/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/baseline/teacher_"+races[split_id[sample]]+extra_img_path[sample], img[sample]) 

    logging.info("Baseline data split completed for %s samples.", cfg.ethnicity)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch margin penalty loss  training')
    args_ = parser.parse_args()
    main(args_)