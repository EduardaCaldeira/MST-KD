import argparse
import logging
import os
import time
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from utils import losses
from config.config import config as cfg
from utils.dataset import MXFaceDataset, MXFaceDataset
from utils.utils_callbacks import CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging

from backbones.iresnet import fc_mapping

torch.backends.cudnn.benchmark = True

def main(args):
    local_rank = args.local_rank
    torch.cuda.set_device(0)
    rank = 0 
    world_size = 1

    # check if the saving path (logging) exists and create it if not
    if not os.path.exists(cfg.out_arc_adapter) and rank == 0:
        os.makedirs(cfg.out_arc_adapter)
    else:
        time.sleep(2)

    # important for logging intialization
    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.out_arc_adapter, logfile="EArcFace_adapter_train.log")

    # trainset initialization (it is the same for training and for saving the embeddings in the new space with the frozen network)
    trainset = MXFaceDataset(root_dir=cfg.rec, ethnicity="All", local_rank=local_rank, is_train=True, to_sample=28000 - 3)
    trainset_no_drop = MXFaceDataset(root_dir=cfg.rec, ethnicity="All", local_rank=local_rank, is_train=False, to_sample=28000 - 3)

    # the train ssamples is also shared
    train_sampler = torch.utils.data.RandomSampler(trainset)
    train_sampler_no_drop = torch.utils.data.RandomSampler(trainset_no_drop)

    # two dataloaders are needed because in the phase where the embeddings are saved we don't want to drop the last samples to have
    # only complete batches
    train_loader = DataLoader(dataset=trainset, batch_size=cfg.batch_size,
        sampler = train_sampler, num_workers=4, pin_memory=True, drop_last=True, prefetch_factor=2)
    train_loader_no_drop = DataLoader(dataset=trainset_no_drop, batch_size=cfg.batch_size,
        sampler = train_sampler_no_drop, num_workers=4, pin_memory=True, drop_last=False, prefetch_factor=2)
    
    # select the appropriate backbone according to the considered method
    if cfg.arc_method=="fusion" or cfg.arc_method=="positioning":
        backbone=fc_mapping(in_features=4*cfg.embedding_size, out_features=cfg.embedding_size).to(local_rank)
    elif cfg.arc_method=="encoding":
        backbone=fc_mapping(in_features=cfg.embedding_size+1, out_features=cfg.embedding_size).to(local_rank)
    else: 
        logging.info("The method for EArcFace (fusion, positioning or encoding) is incorrectly selected!")

    backbone.train()

    header = losses.ElasticArcFace(in_features=cfg.embedding_size, out_features=28000 - 3, s=cfg.s, m=cfg.m).to(local_rank)
    header.train()

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr_fc / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)
    opt_header = torch.optim.SGD(
        params=[{'params': header.parameters()}],
        lr=cfg.lr_fc / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.lr_func)  
    scheduler_header = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_header, lr_lambda=cfg.lr_func)   

    criterion = CrossEntropyLoss()

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch_fc)
    if rank == 0: logging.info("Total Step is: %d" % total_step)

    callback_logging = CallBackLogging(500, rank, total_step, cfg.batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.out_arc_adapter)

    loss = AverageMeter()
    global_step = cfg.global_step

    logging.info("Train FC layer mapping with EArcFace (%d ids).", 28000 - 3)

    races=["Asian", "African", "Caucasian", "Indian"]
   
    # we will save the minimum loss reached to decide which version of the model will be used to extract the final features
    min_loss=math.inf 
    best_step=-1
    for epoch in range(start_epoch, cfg.num_epoch_fc):
        running_loss = 0 
        steps = 0
        for _, (_, label, extra_img_path) in enumerate(train_loader):
            label = label.cuda(local_rank, non_blocking=True)
            global_step += 1

            if cfg.arc_method=="fusion" or cfg.arc_method=="positioning":
                original_emb=np.zeros((len(extra_img_path), 4*cfg.embedding_size))
                for sample in range(len(extra_img_path)):
                    sample_race=extra_img_path[sample].split("/")[1]

                    four_embs=np.empty((1,0))
                    for race in range(len(races)):
                        if races[race]==sample_race:
                            new_emb=np.load(cfg.out_fts+extra_img_path[sample]+".npy").reshape(-1, cfg.embedding_size)
                        else:
                            if cfg.arc_method=="fusion":
                                new_emb=np.load(cfg.out_fts+extra_img_path[sample].replace(sample_race, races[race]+"_"+sample_race)+".npy").reshape(-1, cfg.embedding_size)
                            else:
                                new_emb=np.zeros((1, cfg.embedding_size))
                        four_embs=np.concatenate((four_embs, new_emb), axis=1)
                    original_emb[sample, :]=four_embs
            else:
                original_emb=np.zeros((len(extra_img_path), cfg.embedding_size+1))
                for sample in range(len(extra_img_path)):
                    sample_race=extra_img_path[sample].split("/")[1]
                    original_emb[sample, 0] = races.index(sample_race)
                    original_emb[sample, 1:] = np.load(cfg.out_fts+extra_img_path[sample]+".npy").reshape(-1, cfg.embedding_size)

            features = F.normalize(backbone(torch.from_numpy(original_emb).float().cuda(local_rank, non_blocking=True)))
            thetas = header(features, label)

            # backpropagation and optimization
            loss_v = criterion(thetas, label)
            loss_v.backward()
            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)

            opt_backbone.step()
            opt_header.step()

            opt_backbone.zero_grad()
            opt_header.zero_grad()

            # loss update
            loss.update(loss_v.item(), 1)
            running_loss += loss_v.item()
            steps+=1
            callback_logging(global_step, loss, epoch)
   
        logging.info("Loss check: %f.", running_loss)
        
        # at the end of each epoch, we save its information if it was the best one so far (where the lowest loss was achieved)
        if running_loss < min_loss:
            min_loss=running_loss
            best_step=global_step

        scheduler_backbone.step()
        scheduler_header.step()
        callback_checkpoint(global_step, backbone, header=None) 

    logging.info("EArcFace FC train complete! The best model was %d.", best_step)
 
    # initializating the backbone for evaluation (EArcFace loss)
    if cfg.arc_method=="fusion" or cfg.arc_method=="positioning":
        backbone_eval=fc_mapping(in_features=4*cfg.embedding_size, out_features=cfg.embedding_size).to(local_rank)
    elif cfg.arc_method=="encoding":
        backbone_eval=fc_mapping(in_features=cfg.embedding_size+1, out_features=cfg.embedding_size).to(local_rank)

    # load the weights of the already trained FC layer (backbone)
    try:
        backbone_pth=os.path.join(cfg.out_arc_adapter+"/"+str(best_step)+"backbone.pth")
        backbone_eval.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))

        if rank == 0:
            logging.info("Best EArcFace FC layer for evaluation loaded successfully!")
    except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        logging.info("Load best EArcFace FC layer failed!")

    # set the backbone to evaluation mode and use it to generate the transformed embeddings
    backbone_eval.eval()
    for _, (_, _, extra_img_path) in enumerate(train_loader_no_drop):
        if cfg.arc_method=="fusion" or cfg.arc_method=="positioning":
            original_emb=np.zeros((len(extra_img_path), 4*cfg.embedding_size))
            for sample in range(len(extra_img_path)):
                sample_race=extra_img_path[sample].split("/")[1]

                four_embs=np.empty((1,0))
                for race in range(len(races)):
                    if races[race]==sample_race:
                        new_emb=np.load(cfg.out_fts+extra_img_path[sample]+".npy").reshape(-1, cfg.embedding_size)
                    else:
                        if cfg.arc_method=="fusion":
                            new_emb=np.load(cfg.out_fts+extra_img_path[sample].replace(sample_race, races[race]+"_"+sample_race)+".npy").reshape(-1, cfg.embedding_size)
                        else:
                            new_emb=np.zeros((1, cfg.embedding_size))
                    four_embs=np.concatenate((four_embs, new_emb), axis=1)
                original_emb[sample, :]=four_embs
        else:
            original_emb=np.zeros((len(extra_img_path), cfg.embedding_size+1))
            for sample in range(len(extra_img_path)):
                sample_race=extra_img_path[sample].split("/")[1]
                original_emb[sample, 0] = races.index(sample_race)
                original_emb[sample, 1:] = np.load(cfg.out_fts+extra_img_path[sample]+".npy").reshape(-1, cfg.embedding_size)

        # determine the already trained FC layer's (backbone) output for each embedding => the header is disregarded
        features = F.normalize(backbone_eval(torch.from_numpy(original_emb).float().cuda(local_rank, non_blocking=True)))
        features=features.cpu().detach().numpy()

        for j in range(len(extra_img_path)):
            # check if the path where the features will be saved exists and create it if not
            if not os.path.exists(cfg.out_arc_fts+extra_img_path[j].replace(extra_img_path[j].split("/")[3],"")[0:-1]):
                os.makedirs(cfg.out_arc_fts+extra_img_path[j].replace(extra_img_path[j].split("/")[3],"")[0:-1])

            # save each embedding in an appropriate folder
            np.save(cfg.out_arc_fts+extra_img_path[j], features[j,:])

    logging.info("Features extracted successfully after applying the FC layer!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--resume', type=int, default=0, help="resume training")
    args_ = parser.parse_args()
    main(args_)