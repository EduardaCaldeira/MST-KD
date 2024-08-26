import argparse
import logging
import os
import time
import torch
import torch.nn.functional as F
import torch.utils.data.distributed
import numpy as np
from torch.utils.data import DataLoader
from config.config import config as cfg
from utils.dataset import MXFaceDataset, MXFaceDataset
from utils.utils_logging import init_logging
from backbones.iresnet import iresnet100, iresnet50, iresnet34

torch.backends.cudnn.benchmark = True

def main(args):
    local_rank=args.local_rank
    torch.cuda.set_device(local_rank)
    rank=0

    # check if the path where the features will be saved exists and create it if not
    if not os.path.exists(cfg.out_fts) and rank == 0:
        os.makedirs(cfg.out_fts)
    else:
        time.sleep(2)

    # important for the outputted log file
    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output, logfile="emb_extraction.log")

    # we want to extract all the samples, regardless of the considered ethnicity
    trainset = MXFaceDataset(root_dir=cfg.rec, ethnicity="All", local_rank=local_rank, is_train=False, to_sample=28000-3)

    # initilization of the sampler and loader; 'drop_last' should be set to false to extract all the embeddings
    train_sampler = torch.utils.data.RandomSampler(trainset)
    train_loader = DataLoader(dataset=trainset, batch_size=cfg.batch_size,
        sampler = train_sampler, num_workers=4, pin_memory=True, drop_last=False, prefetch_factor=2) 
    
    # load model
    if cfg.network == "iresnet100":
        backbone = iresnet100(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
        # FIXME Verify dropout rate
    elif cfg.network == "iresnet50":
        backbone = iresnet50(dropout=0.4,num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "iresnet34":
        backbone = iresnet34(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    else:
        backbone = None
        logging.info("load backbone failed!")
        exit()

    # load the weights of the already trained model
    try:
        backbone_pth=os.path.join(cfg.backbone_pth)
        backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))

        if rank == 0:
            logging.info("backbone resume loaded successfully!")
    except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        logging.info("load backbone resume init, failed!")

    # set the backbone to evaluation mode
    backbone.eval()

    for _, (img, label, extra_img_path) in enumerate(train_loader):
        # assign image and label information to cuda
        img=img.cuda(local_rank, non_blocking=True)
        label=label.cuda(local_rank, non_blocking=True)

        # extract the features from the images
        features=F.normalize(backbone(img))
        features=features.cpu().detach().numpy()

        for j in range(label.size()[0]):
            sample_race=extra_img_path[j].split("/")[1]
            if sample_race!=cfg.ethnicity: # the features of the same ethnicity were already extracted
                # the directory follows the organization "ethnicityOfExtractor_ethnicityOfSample"
                extra_img_path[j]=extra_img_path[j].replace(sample_race, cfg.ethnicity+"_"+sample_race)

            # check if the path where the features will be saved exists and create it if not
            if not os.path.exists(cfg.out_fts+extra_img_path[j].replace(extra_img_path[j].split("/")[3],"")[0:-1]):
                os.makedirs(cfg.out_fts+extra_img_path[j].replace(extra_img_path[j].split("/")[3],"")[0:-1])

            # save each embedding in an appropriate folder
            np.save(cfg.out_fts+extra_img_path[j], features[j,:])

    logging.info("Features from different races extracted successfully for %s!", cfg.ethnicity)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch margin penalty loss training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    args_ = parser.parse_args()
    main(args_)