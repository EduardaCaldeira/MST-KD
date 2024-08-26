import logging
import os
import time
from typing import List
# FIXME
import wandb
import torch

from eval import verification
from utils.utils_logging import AverageMeter


class CallBackVerification(object):
    def __init__(self, frequent, rank, val_targets, rec_prefix, image_size=(112, 112)):
        self.frequent: int = frequent
        self.rank: int = rank
        self.highest_acc: float = 0.0
        self.best_threshold: List[float] = [0.0] * len(val_targets)
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.val_targets = val_targets
        if self.rank == 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)
        

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list,thrs = verification.test(
                self.ver_list[i], backbone, 10, 10)
            logging.info('[%s][%d]Thrs: %f' % (self.ver_name_list[i], global_step, thrs))
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            # FIXME
            """ 
            try:
                wandb.log({"acc_" + self.val_targets[i]  : acc2})
            except:
                wandb.init(
                # set the wandb project where this run will be logged
                project="BalancedFace",
                #track hyperparameters and run metadata
               config={
                "count_for_images": True,
                 "test": True
                }
            ) 
            """
            if acc2 > self.highest_acc_list[i]:
                self.best_threshold[i] = thrs
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            logging.info(
                '[%s][%d]Thrs-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.best_threshold[i]))

            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join("/nas-ctm01/datasets/public/BIOMETRICS/Face_Recognition/faces_emore", name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(  path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank == 0 :#and num_update > 0 and num_update % self.frequent == 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


class CallBackLogging(object):
    def __init__(self, frequent, rank, total_step, batch_size, world_size, writer=None, resume=0, rem_total_steps=None):
        self.frequent: int = frequent
        self.rank: int = rank
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.world_size: int = world_size
        self.writer = writer
        self.resume = resume
        self.rem_total_steps = rem_total_steps

        self.init = False
        self.tic = 0

    def __call__(self, global_step, loss: AverageMeter, epoch: int):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                time_now = (time.time() - self.time_start) / 3600
                # TODO: resume time_total is not working
                if self.resume:
                    time_total = time_now / ((global_step + 1) / self.rem_total_steps)
                else:
                    time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                msg = "Speed %.2f samples/sec   Loss %.4f Epoch: %d   Global Step: %d   Required: %1.f hours" % (
                    speed_total, loss.avg, epoch, global_step, time_for_end
                )
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()

class CallBackModelCheckpoint(object):
    def __init__(self, rank, output="./"):
        self.rank: int = rank
        self.output: str = output

    def __call__(self, global_step, backbone: torch.nn.Module, header: torch.nn.Module = None, ethnicity = None):
        if ethnicity is not None:
            if global_step > 100 and self.rank == 0:
                torch.save(backbone.state_dict(), os.path.join(self.output, str(ethnicity)+"_"+str(global_step)+"backbone.pth"))
            if global_step > 100 and header is not None:
                torch.save(header.state_dict(), os.path.join(self.output, str(ethnicity)+"_"+str(global_step)+ "header.pth"))
        else:
            if global_step > 100 and self.rank == 0:
                torch.save(backbone.state_dict(), os.path.join(self.output, str(global_step)+ "backbone.pth"))
            if global_step > 100 and header is not None:
                torch.save(header.state_dict(), os.path.join(self.output, str(global_step)+ "header.pth"))

class CallBackAdaptorTest(object):
    def __init__(self, frequent, rank, val_targets, rec_prefix, image_size=(112, 112)):
        self.frequent: int = frequent
        self.rank: int = rank
        self.highest_acc: float = 0.0
        self.best_threshold: List[float] = [0.0] * len(val_targets)
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.val_targets = val_targets
        if self.rank == 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)
        
    def ver_test(self, backbone_African, backbone_Asian, backbone_Caucasian, backbone_Indian, adaptor, global_step, method):
        results = []
        for i in range(len(self.ver_list)):
            _, _, acc2, std2, xnorm, _, thrs = verification.adaptor_test(self.ver_list[i], backbone_African, backbone_Asian, backbone_Caucasian, backbone_Indian, adaptor, method, 10, 10)
            logging.info('[%s][%d]Thrs: %f' % (self.ver_name_list[i], global_step, thrs))
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            
            if acc2 > self.highest_acc_list[i]:
                self.best_threshold[i] = thrs
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            logging.info(
                '[%s][%d]Thrs-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.best_threshold[i]))
            
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join("/nas-ctm01/datasets/public/BIOMETRICS/Face_Recognition/faces_emore", name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone_African, backbone_Asian, backbone_Caucasian, backbone_Indian, adaptor, method=None):
        if self.rank == 0:
            backbone_African.eval()
            backbone_Asian.eval()
            backbone_Caucasian.eval()
            backbone_Indian.eval()
            adaptor.eval()
            self.ver_test(backbone_African, backbone_Asian, backbone_Caucasian, backbone_Indian, adaptor, num_update, method)