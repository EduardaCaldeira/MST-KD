import numbers
import os
import queue as Queue
import threading

import random
import math
import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from glob import glob
import wandb
# FIXME
os.environ["WANDB__SERVICE_WAIT"] = "300"

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch



class MXFaceDataset_rec(Dataset):
    def __init__(self, root_dir, local_rank,transform):
        super(MXFaceDataset_rec, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)

        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

        wandb.init(
            # set the wandb project where this run will be logged
            project="BalancedFace",

            # track hyperparameters and run metadata
            config={
                "count_for_images": True
            }
        )

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label,0

    def __len__(self):
        return len(self.imgidx)


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, ethnicity, local_rank, is_train=True, to_sample=0):
        super(MXFaceDataset, self).__init__()

        # no horizontal flip should be performed when the embeddings are being extracted
        if is_train:
            self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
        else:
            self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
            
        self.root_dir = root_dir
        self.local_rank = local_rank
        self.image_paths = glob(root_dir+"/*/*/*")
        new_image_paths = []
        self.labels = {}
        current_key = 0
       
        # ========================================================================================
        # Eduarda: modifications done by me   

        self.ethnicity = ethnicity
        self.to_sample = to_sample

        if self.ethnicity=="All":
            if self.to_sample<28000-3:
                self.protocol_random_balanced(new_image_paths, to_sample)
            else:
                self.protocol_select_all(new_image_paths)
        else:
            self.protocol_select_ethnicity(new_image_paths)

        # ========================================================================================

        self.probs = {}
        for path in self.image_paths: 
            label = path.replace(root_dir,"").split("/")[2]
            if  label in self.labels: 
                continue
            else:
                self.labels[label] = current_key
                try:
                    self.probs[label] = (-1 * (mean_caucasians - caucasian_samples[label]) + 1)**2 #((mean_caucasians - caucasian_samples[label]) + 1)**2 #
                except:
                   try:
                       self.probs[label] = (-1 * (mean_asians - asian_samples[label]) + 1)**2 # ((mean_asians - asian_samples[label]) + 1)**2 #
                   except:
                    try:
                        self.probs[label] = (-1 *(mean_indians - indian_samples[label]) + 1)**2 #((mean_indians - indian_samples[label]) + 1)**2 #
                    except:
                        try:
                            self.probs[label] = (-1 * (mean_africans - african_samples[label]) + 1)**2 #((mean_africans - african_samples[label]) + 1)**2 #
                        except:
                            self.probs[label] = 1.
                current_key+=1
        
        print(current_key)

    # ========================================================================================
    # Eduarda: modifications done by me  
        
    def protocol_select_all(self, new_image_paths):
        for path in self.image_paths:
            new_image_paths.append(path)
        self.image_paths = new_image_paths
        
    def protocol_select_ethnicity(self, new_image_paths):
        for path in self.image_paths:
            if path.replace(self.root_dir, "").split("/")[1] == self.ethnicity:
                new_image_paths.append(path)
        self.image_paths = new_image_paths

    def protocol_random_balanced(self, new_image_paths, to_sample):
        for race in ["African", "Caucasian", "Asian", "Indian"]:
            labels = []
            print("Sampling " + race)
            for path in self.image_paths:
                if path.replace(self.root_dir, "").split("/")[1] == race:
                    labels.append(race + "/" + path.replace(self.root_dir, "").split("/")[2])
            labels = list(set(labels))
            samples_to_keep = np.random.choice(labels, to_sample // 4, replace=False)

            for path in self.image_paths:
                if path.replace(self.root_dir, "").split("/")[1] == race:
                    if race + "/" + path.replace(self.root_dir, "").split("/")[2] in samples_to_keep:
                        new_image_paths.append(path)
            print("Sampling of " + race + " samples finished")
        
        self.image_paths = new_image_paths

    # ========================================================================================

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        image_id = image_path.replace(self.root_dir,"")
        label = self.labels[image_id.split("/")[2]]
        prob = self.probs[image_id.split("/")[2]]

        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.imread(image_path)
        sample = cv2.cvtColor(sample,cv2.COLOR_BGR2RGB)

        # ========================================================================================
        # Eduarda: modifications done by me
        extra_img_path = image_path.replace(self.root_dir,"").replace(".jpg","")
        # ========================================================================================

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, extra_img_path

    def __len__(self):
        return len(self.image_paths)

# ========================================================================================
# Eduarda: modifications done by me
class MXFaceDatasetSplit(Dataset):
    def __init__(self, root_dir, ethnicity):
        super(MXFaceDatasetSplit, self).__init__()

        self.root_dir = root_dir
        self.image_paths = glob(root_dir+"/*/*/*")
        self.ids = []
        new_image_paths = []
        self.split_id = []
        self.ethnicity = ethnicity

        if self.ethnicity=="All":
            exit("Please select a specific ethnicity!")

        self.protocol_select_ethnicity(new_image_paths)
        self.protocol_rand_split()

    def protocol_select_ethnicity(self, new_image_paths):
        self.image_ids = glob(self.root_dir+"/"+self.ethnicity+"/*")
        for path in self.image_paths:
            if path.replace(self.root_dir, "").split("/")[1] == self.ethnicity:
                new_image_paths.append(path)
        self.image_paths = new_image_paths 
        
    def protocol_rand_split(self):
        labels=self.image_ids.copy()
        random.shuffle(labels)

        lim1 = math.ceil(len(labels)/4)
        lim2 = lim1+math.ceil((len(labels)-lim1)/3)
        lim3 = lim2+math.ceil((len(labels)-lim2)/2)         

        for path in self.image_paths:
            if self.image_ids.index((path.replace(path.replace(self.root_dir, "").split("/")[3],""))[0:-1]) < lim1:
                self.split_id.append(0)
            elif self.image_ids.index((path.replace(path.replace(self.root_dir, "").split("/")[3],""))[0:-1]) < lim2:
                self.split_id.append(1)
            elif self.image_ids.index((path.replace(path.replace(self.root_dir, "").split("/")[3],""))[0:-1]) < lim3:
                self.split_id.append(2)
            else:
                self.split_id.append(3)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        split_id = self.split_id[index]
        extra_img_path = image_path.replace(self.root_dir,"")

        sample = cv2.imread(image_path)
        return sample, split_id, extra_img_path
    
    def __len__(self):
        return len(self.image_paths)
# ========================================================================================   

class MXSyntheticFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank,from_file = "",transform=None):
        super(MXSyntheticFaceDataset, self).__init__()
        if transform is None:
            self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                 ])
        else:
            self.transform = transform
        self.root_dir = root_dir
        self.local_rank = local_rank
        self.image_paths = glob(root_dir+"/*/*")
        if from_file != "":
            self.image_paths = []
            with open(from_file, "r") as f:
                image_paths = f.readlines()
            for path in image_paths:
                self.image_paths.append(path.replace("\n",""))

        new_image_paths = []
        self.labels = {}
        current_key = 0
        to_sample = 10721
        # FIXME
        wandb.init(
                # set the wandb project where this run will be logged
                project="BalancedFace",
                # track hyperparameters and run metadata
                config={
                "count_for_images": True,
                "sampled": to_sample,
                "ignore_identities":False,
                "unbalance": True,
                "force":True
                }
            )
        self.probs = {}
        for path in self.image_paths: 
            label = os.path.join(*path.split("/")[:-1])
            if  label in self.labels: 
                continue
            else:
                self.labels[label] = current_key
                try:
                    self.probs[label] = (-1 * (mean_caucasians - caucasian_samples[label]) + 1)**2 #((mean_caucasians - caucasian_samples[label]) + 1)**2 #
                except:
                   try:
                       self.probs[label] = (-1 * (mean_asians - asian_samples[label]) + 1)**2 # ((mean_asians - asian_samples[label]) + 1)**2 #
                   except:
                    try:
                        self.probs[label] = (-1 *(mean_indians - indian_samples[label]) + 1)**2 #((mean_indians - indian_samples[label]) + 1)**2 #
                    except:
                        try:
                            self.probs[label] = (-1 * (mean_africans - african_samples[label]) + 1)**2 #((mean_africans - african_samples[label]) + 1)**2 #
                        except:
                            #print("key not found")
                            #print(label)
                            self.probs[label] = 1.
                current_key+=1
        print(current_key)
        

    def __getitem__(self, index):
        p = np.random.uniform(0,1,1)
        image_path = self.image_paths[index]
        image_id = image_path
        if p > 0.5: 
             image_path = self.image_paths[index].replace("ca-cpd25-synthetic-uniform-10050/ca-cpd25-synthetic-uniform-10050","occluded_ca-cpd25-synthetic-uniform-10050/Protocol4/OCC").replace("ca-cpd25-synthetic-two-stage-10050_","occluded_ca-cpd25-synthetic-two-stage-10050_/Protocol4/OCC")
        #image_id = image_path
        label = self.labels[os.path.join(*image_id.split("/")[:-1])]
        prob = self.probs[os.path.join(*image_id.split("/")[:-1])]
        #mx.image.imdecode(img).asnumpy()
        #idx = self.imgidx[index]
        #s = self.imgrec.read_idx(idx)
        #header, img = mx.recordio.unpack(s)
        #label = header.label
        #if not isinstance(label, numbers.Number):
        #    label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.imread(image_path)
        sample = cv2.cvtColor(sample,cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, prob

    def __len__(self):
        return len(self.image_paths)

def remove_from_dict(keys, dict):
    for key in keys:
        del dict[key]
    return dict 

class FaceDatasetFolder(Dataset):
    def __init__(self, root_dir, local_rank):
        super(FaceDatasetFolder, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        self.imgidx, self.labels=self.scan(root_dir)
    def scan(self,root):
        imgidex=[]
        labels=[]
        lb=-1
        list_dir=os.listdir(root)
        list_dir.sort()
        for l in list_dir:
            images=os.listdir(os.path.join(root,l))
            lb += 1
            for img in images:
                imgidex.append(os.path.join(l,img))
                labels.append(lb)
        return imgidex,labels
    def readImage(self,path):
        return cv2.imread(os.path.join(self.root_dir,path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img=self.readImage(path)
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)
