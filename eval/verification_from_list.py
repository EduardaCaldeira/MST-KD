from os import device_encoding
import sys
sys.path.append('/home/pedro/PycharmProjects/RaceBalancedFaceRecognition')
#python3 eval/verification_from_list.py --model_path=/home/pedro/Desktop/Res50-baseline_casia_curricular/130288backbone.pth --image_list=/home/pedro/Desktop/sub-tasks_1.1_1.2_1.3/image_list.txt --save_path="./baseline/" --list_pairs=/home/pedro/Desktop/sub-tasks_1.1_1.2_1.3/bupt_comparison.txt --output_file="baseline.txt"
# casia thr = 1.32000

#python3 eval/verification_from_list.py --model_path=/home/pedro/Desktop/R50_ElasticArcFace_two_stage_rand_augment/130288backbone.pth --image_list=/home/pedro/Desktop/sub-tasks_1.1_1.2_1.3/image_list.txt --save_path="./baseline/" --list_pairs=/home/pedro/Desktop/sub-tasks_1.1_1.2_1.3/bupt_comparison.txt --output_file="synth_baseline.txt"

# TODO Normalize embeddings
# TODO Flip embedding
import torch
import imageio
import torch.nn as nn
import cv2
from backbones.iresnet import iresnet50
from tqdm import tqdm
import numpy as np
import os
import sklearn
from sklearn.preprocessing import normalize
import argparse
from  sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

parser = argparse.ArgumentParser(description='OCFR script')

parser.add_argument('--device', default='0', help='gpu id')
parser.add_argument('--model_path', default='../../elastic_arc_plus.pth', help='path to pretrained model')
parser.add_argument('--image_list', type=str, default='',help='pairs file')

parser.add_argument('--list_pairs', type=str, default='',help='pairs file')
parser.add_argument('--data_path', type=str, default='',help='root path to data')

parser.add_argument('--save_path', type=str, default='',help='root path to data')

parser.add_argument('--output_file', type=str, default='',help='final path path to resu')

args = parser.parse_args()

RESULTS_FILE = args.output_file


class FaceModel():
    def __init__(self, model_path, save_path,image_list,data_path,ctx_id):
        self.ctx_id=ctx_id
        self.model=self._get_model(model_path,ctx_id)
        self.save_path=save_path
        if not(os.path.isdir(self.save_path)):
            os.makedirs(save_path)
        self.image_list=image_list
        self.data_path=data_path
        self.k = 0
        self.k2 = 0
        self.failed = []
    def _get_model(self, model_path,ctx_id):
        pass

    def _getFeatureBlob(self,input_blob):
        pass
    def read(self,image_path):
        try:
            img = cv2.imread(image_path)
        except:
            self.k+=1
            print(self.k)
        return img
    def save(self,features,image_path_list,alignment_results):
        # Save embedding as numpy to disk in save_path folder
        for i in tqdm(range(len(features))):
            filename = str(str(image_path_list[i]).replace("/",".").replace("\n",""))
            np.save(os.path.join(self.save_path, filename), features[i])
        np.save(os.path.join(self.save_path,"alignment_results.txt"),np.asarray(alignment_results))

    def process(self,image,bbox):
        #image = image[int(float(bbox[1])):int(float(bbox[3])), int(float(bbox[0])):int(float(bbox[2]))]

        nimg =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        nimg= cv2.resize(nimg, (112,112))
        nimg2 = cv2.flip(nimg, 1)
        return np.asarray([nimg,nimg2]), "S"

    def distance(self,embedding1, embedding2):
        return euclidean_distances(sklearn.preprocessing.normalize(embedding1.reshape((1,-1))),sklearn.preprocessing.normalize(embedding2.reshape((1,-1))))

    def save_score(self,score):
        with open(RESULTS_FILE,"a") as f:
            f.write(str(score[0][0])+"\n")

    def comparison(self,list_pairs):
        with open(list_pairs, "r") as f:
            for line in f:
                ##import pdb
                ##pdb.set_trace()
                emb1=np.load(os.path.join(self.save_path,line.split(";")[0].replace("/",".").replace("\n","")+".npy")) #.replace(".jpg","")
                emb2=np.load(os.path.join(self.save_path,line.split(";")[1].replace("/",".").replace("\n","")+".npy"))
                score=self.distance(emb1,emb2)
                self.save_score(score)

    def read_img_path_list(self):
        with open(self.image_list, "r") as f:
            lines = f.readlines()
            file_path = [os.path.join(self.data_path, line.rstrip().split()[0]) for line in lines]
            bbx = [line.rstrip().split()[1:] for line in lines]
        return file_path ,bbx


    def get_batch_feature(self, image_path_list, bbx, batch_size=64, flip=0):

        count = 0
        num_batch =  int(len(image_path_list) / batch_size)
        features = []
        alignment_results = []
        for i in range(0, len(image_path_list), batch_size):
            #print(count)
            if count < num_batch:
                tmp_list = image_path_list[i : i+batch_size]
                tmp_list_bbx = bbx[i : i+batch_size]

            else:
                tmp_list = image_path_list[i :]
                tmp_list_bbx = bbx[i :]

            count += 1

            images = []
            for i  in range(len(tmp_list)):
                self.k2+=1
                image_path=tmp_list[i]
                bbox=tmp_list_bbx[i]
                image=self.read("/home/pedro/Desktop/race_per_7000_aligned/"+image_path)
                try:
                    image,alignment_result=self.process(image,bbox)
                except Exception as e:
                    image = self.read("/home/pedro/Desktop/race_per_7000/"+image_path)
                    image, alignment_result = self.process(image, bbox)
                    self.k+=1
                    self.failed.append(image_path)
                    print(self.k)
                    print(self.k2)
                    #continue
                alignment_results.append(alignment_result)
                images.append(image)
            input_blob = np.array(images)

            emb = self._getFeatureBlob(input_blob)
            features.append(emb)
        print(self.failed)
        features = np.vstack(features)
        self.save(features,image_path_list,alignment_results)
        return


class ElasticFaceModel(FaceModel):
    def __init__(self, model_path, save_path,image_list,data_path,ctx_id):
        super(ElasticFaceModel, self).__init__( model_path, save_path,image_list,data_path,ctx_id)

    def _get_model(self, model_path, ctx_id):
        weight = torch.load(os.path.join(model_path))
        backbone = iresnet50().to(f"cuda:{ctx_id}")
        backbone.load_state_dict(weight)
        model = backbone.to("cuda:"+str(ctx_id))
        model.eval()
        return model

    @torch.no_grad()
    def _getFeatureBlob(self,input_blob):
        input_blob2 = np.transpose(input_blob[:,1,:,:,:], (0,3,1,2))
        input_blob = np.transpose(input_blob[:,0,:,:,:], (0,3,1,2))


        imgs = torch.Tensor(input_blob).to("cuda:"+str(self.ctx_id))
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat = self.model(imgs)

        imgs = torch.Tensor(input_blob2).to("cuda:"+str(self.ctx_id))
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat += self.model(imgs)
        #feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()

def main(args):
    model=ElasticFaceModel(args.model_path,args.save_path,args.image_list,args.data_path,args.device)
    file_path, bbx= model.read_img_path_list()
    model.get_batch_feature(file_path, bbx)
    model.comparison(args.list_pairs)
    from pyeer.eer_info import get_eer_stats
    from pyeer.report import generate_eer_report, export_error_rates
    from pyeer.plot import plot_eer_stats

    issame_list = []
    gscores_a = []
    iscores_a = []
    # with open("../../A_A_issame.txt","r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         issame_list.append(int(line))
    #
    # with open(RESULTS_FILE,"r") as f:
    #     lines = f.readlines()
    #     for i,line in enumerate(lines):
    #         if line == "": continue
    #         if issame_list[i] == 1:
    #             gscores_a.append(float(line))
    #         else:
    #             iscores_a.append(float(line))
    #
    # stats_a = get_eer_stats(gscores_a, iscores_a)
    # generate_eer_report([stats_a], ['A'], RESULTS_FILE.replace(".txt",'.html'))

if __name__ == '__main__':
    #args = parser.parse_args()
    main(args)

















