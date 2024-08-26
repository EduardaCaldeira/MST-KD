import numbers
import os
import queue as Queue
import threading

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from glob import glob
import seaborn as sns
from matplotlib import pyplot as plt

class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        self.image_paths = glob(root_dir + "/*/*/*")
        new_image_paths = []
        self.labels = {}
        current_key = 0
        self.to_sample = 1000
        self.legend_dict = {0:"28k" ,1000:"27k", 3500:"24.5k", 7000: "21k", 14000: "14k"}

        #### RANDOM SELECTION
        #self.protocol_random_ethnicity(new_image_paths, root_dir)

        #### MAX SELECTION
        #self.protocol_max_ethnicity(new_image_paths, root_dir)

        ## REDUCED
        #new_image_paths = self.random_balanced(new_image_paths, root_dir, to_sample)

        # # Proposed A
        #self.protocol_a(new_image_paths, root_dir)

        #### *Proposed A - With relabelling

        #self.protocol_a_relabel(new_image_paths, root_dir)

        # Proposed D
        #self.protocol_d(new_image_paths, root_dir)

        # PROPOSED E
        self.protocol_e(new_image_paths, root_dir)

        ### Strategy E with Relabel
        #self.protocol_e_relabel(new_image_paths, root_dir, to_sample)

        ### Strategy D with Relabel
        #self.protocol_d_relabel(new_image_paths, root_dir)

        #### *Proposed 3 - With multiple-forces (Ignoring Identities)
        #african_samples, asian_samples, caucasian_samples, indian_samples, mean_africans, mean_asians, mean_caucasians, mean_indians = self.protocol_multiple_forces(
        #    new_image_paths, root_dir, to_sample)

        ### Random Unbalanced
        #self.protocol_random(root_dir, to_sample)

        ## NEXT
        self.probs = {}
        for path in self.image_paths:
            label = path.replace(root_dir, "").split("/")[2]
            if label in self.labels:
                continue
            else:
                self.labels[label] = current_key
                try:
                    self.probs[label] = (-1 * (mean_caucasians - caucasian_samples[
                        label]) + 1) ** 2  # ((mean_caucasians - caucasian_samples[label]) + 1)**2 #
                except:
                    try:
                        self.probs[label] = (-1 * (mean_asians - asian_samples[
                            label]) + 1) ** 2  # ((mean_asians - asian_samples[label]) + 1)**2 #
                    except:
                        try:
                            self.probs[label] = (-1 * (mean_indians - indian_samples[
                                label]) + 1) ** 2  # ((mean_indians - indian_samples[label]) + 1)**2 #
                        except:
                            try:
                                self.probs[label] = (-1 * (mean_africans - african_samples[
                                    label]) + 1) ** 2  # ((mean_africans - african_samples[label]) + 1)**2 #
                            except:
                                # print("key not found")
                                # print(label)
                                self.probs[label] = 1.
                current_key += 1

        print(current_key)

    def protocol_random_ethnicity(self, new_image_paths, root_dir):
        ethnicity = "Caucasian"
        labels = []
        for path in self.image_paths:
            if path.replace(root_dir, "").split("/")[1] == ethnicity:
                labels.append(ethnicity + "/" + path.replace(root_dir, "").split("/")[2])
        labels = list(set(labels))
        samples_to_remove = np.random.choice(labels, 3500, replace=False)
        for path in self.image_paths:
            if path.replace(root_dir, "").split("/")[1] == ethnicity:
                if ethnicity + "/" + path.replace(root_dir, "").split("/")[2] in samples_to_remove:
                    continue
            new_image_paths.append(path)
        self.image_paths = new_image_paths

    def protocol_max_ethnicity(self, new_image_paths, root_dir):
        ethnicity = "Indian"
        with open('/Users/pedroneto/Desktop/ethnicity scores/predictions.txt') as f:
            predictions = f.readlines()
        ethnicities = {"Caucasian": 0, "Asian": 1, "Indian": 2, "African": 3}
        predictions = [line.split(",") for line in predictions if ethnicity in line]
        new_predictions = {}
        for prediction in predictions:
            label = prediction[5].replace(root_dir, "").split("/")[2]
            if label in new_predictions:
                new_predictions[label].append(float(prediction[ethnicities[ethnicity]]))
            else:
                new_predictions[label] = [float(prediction[ethnicities[ethnicity]])]
        mean_predictions = {}
        for key in new_predictions.keys():
            mean_predictions[key] = sum(d for d in new_predictions[key]) / len(new_predictions[key])
        values, keys = (list(t) for t in zip(*sorted(zip(mean_predictions.values(), mean_predictions.keys()))))
        # for max
        samples_to_remove = keys[:3500]
        # for min
        # samples_to_remove = keys[3499:]
        for path in self.image_paths:
            if path.replace(root_dir, "").split("/")[1] == ethnicity:
                if path.replace(root_dir, "").split("/")[2] in samples_to_remove:
                    continue
            new_image_paths.append(path)
        self.image_paths = new_image_paths

    def random_balanced(self, new_image_paths, root_dir, to_sample):
        labels = []
        for race in ["African", "Caucasian", "Asian", "Indian"]:
            labels = []
            new_image_paths = []
            print("Starting " + race)
            for path in self.image_paths:
                if path.replace(root_dir, "").split("/")[1] == race:
                    labels.append(race + "/" + path.replace(root_dir, "").split("/")[2])
            labels = list(set(labels))
            samples_to_remove = np.random.choice(labels, to_sample // 4, replace=False)

            for path in self.image_paths:
                if path.replace(root_dir, "").split("/")[1] == race:
                    if race + "/" + path.replace(root_dir, "").split("/")[2] in samples_to_remove:
                        continue
                new_image_paths.append(path)
            print("Ending " + race)
            self.image_paths = new_image_paths
        return new_image_paths

    def protocol_a(self, new_image_paths, root_dir):
        african_samples = {}
        caucasian_samples = {}
        asian_samples = {}
        indian_samples = {}
        for base in ["African", "Caucasian", "Asian", "Indian"]:
            print(base)
            with open('/Users/pedroneto/Desktop/ethnicity scores/predictions.txt') as f:
                predictions = f.readlines()

                predictions = [line.split(",") for line in predictions if base in line]
                new_predictions = {}
                index = 0
                if base == "Caucasian": index = 0
                if base == "Asian": index = 1
                if base == "Indian": index = 2
                if base == "African": index = 3
                for prediction in predictions:
                    label = prediction[5].replace(root_dir, "").split("/")
                    label = label[2]
                    if label in new_predictions:
                        new_predictions[label].append(float(prediction[index]))
                    else:
                        new_predictions[label] = [float(prediction[index])]

                mean_predictions = {}
                for key in new_predictions.keys():
                    mean_predictions[key] = sum(d for d in new_predictions[key]) / len(new_predictions[key])

                if base == "Caucasian": caucasian_samples = mean_predictions.copy()
                if base == "Asian": asian_samples = mean_predictions.copy()
                if base == "Indian": indian_samples = mean_predictions.copy()
                if base == "African": african_samples = mean_predictions.copy()

        samples_removed = []
        indians_removed = 0
        caucasians_removed = 0
        asians_removed = 0
        africans_removed = 0

        values_ca, keys_ca = (list(t) for t in
                              zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
        values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
        values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
        values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))


        fig, axs = plt.subplots(nrows=2,ncols=2,)
        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_af), bw=0.5, ax=axs[0][0],label=self.legend_dict[len(samples_removed)])
        plt.xlim(0, 1)

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_as), bw=0.5, ax=axs[0][1],label=self.legend_dict[len(samples_removed)])
        plt.xlim(0, 1)

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_in), bw=0.5, ax=axs[1][0],label=self.legend_dict[len(samples_removed)])
        plt.xlim(0, 1)

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_ca), bw=0.5, ax=axs[1][1],label=self.legend_dict[len(samples_removed)])
        plt.xlim(0, 1)

        while len(samples_removed) < self.to_sample:

            values_ca, keys_ca = (list(t) for t in
                                  zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
            values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
            values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
            values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))

            if len(samples_removed) in [1000,3500,7000]:
               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_af), bw=0.5, ax=axs[0][0], label=self.legend_dict[len(samples_removed)])

               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_as), bw=0.5, ax=axs[0][1], label=self.legend_dict[len(samples_removed)])

               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_in), bw=0.5, ax=axs[1][0], label=self.legend_dict[len(samples_removed)])

               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_ca), bw=0.5, ax=axs[1][1], label=self.legend_dict[len(samples_removed)])

            remove_step = 1
            if np.mean(values_ca) < np.mean(values_as) and np.mean(values_ca) < np.mean(values_af) and np.mean(
                    values_ca) < np.mean(values_in):
                # print("Removing Caucasian")
                caucasian_samples = remove_from_dict(keys_ca[:remove_step], caucasian_samples)
                samples_removed += keys_ca[:remove_step]
                caucasians_removed += remove_step
                continue

            if np.mean(values_as) < np.mean(values_ca) and np.mean(values_as) < np.mean(values_af) and np.mean(
                    values_as) < np.mean(values_in):
                # print("Removing Asian")
                asian_samples = remove_from_dict(keys_as[:remove_step], asian_samples)
                samples_removed += keys_as[:remove_step]
                asians_removed += remove_step
                continue

            if np.mean(values_in) < np.mean(values_ca) and np.mean(values_in) < np.mean(values_af) and np.mean(
                    values_in) < np.mean(values_as):
                # print("Removing Indian")
                indian_samples = remove_from_dict(keys_in[:remove_step], indian_samples)
                samples_removed += keys_in[:remove_step]
                indians_removed += remove_step
                continue

            # print("Removing African")
            african_samples = remove_from_dict(keys_af[:remove_step], african_samples)
            samples_removed += keys_af[:remove_step]
            africans_removed += remove_step
        values_ca, keys_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
        values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
        values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
        values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))
        print("Means")
        print(np.mean(values_ca))
        print(np.mean(values_as))
        print(np.mean(values_in))
        print(np.mean(values_af))
        print(caucasians_removed)
        print(asians_removed)
        print(indians_removed)
        print(africans_removed)
        print(len(values_ca))
        print(len(values_as))
        print(len(values_in))
        print(len(values_af))

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_af), bw=0.5,ax=axs[0][0], label=self.legend_dict[len(samples_removed)])
        axs[0][0].set_xlim(0.5, 1)
        axs[0][0].set_title("African Distribution")
        axs[0][0].legend()

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_as), bw=0.5,ax=axs[0][1], label=self.legend_dict[len(samples_removed)])
        axs[0][1].set_xlim(0.5, 1)
        axs[0][1].set_title("Asian Distribution")
        axs[0][1].legend()
        
        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_in), bw=0.5,ax=axs[1][0], label=self.legend_dict[len(samples_removed)])
        axs[1][0].set_xlim(0.5, 1)
        axs[1][0].set_title("Indian Distribution")
        axs[1][0].legend()
        
        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_ca), bw=0.5,ax=axs[1][1], label=self.legend_dict[len(samples_removed)])
        axs[1][1].set_xlim(0.5, 1)
        axs[1][1].set_title("Caucasian Distribution")
        axs[1][1].legend()
        plt.tight_layout()
        plt.savefig('../ethnicity_score_plots/protocol_a'+str(self.to_sample)+'.png', dpi=300)
        plt.show()
        mean_africans = np.mean(values_af)
        mean_caucasians = np.mean(values_ca)
        mean_asians = np.mean(values_as)
        mean_indians = np.mean(values_in)


    def protocol_a_relabel(self, new_image_paths, root_dir):
        caucasian_samples = {}
        african_samples = {}
        asian_samples = {}
        indian_samples = {}
        with open('/Users/pedroneto/Desktop/ethnicity scores/predictions.txt') as f:
            predictions = f.readlines()

            predictions = [line.replace(" ", "").split(",") for line in predictions]
            new_predictions = {}
            # import pdb
            # pdb.set_trace()
            for prediction in predictions:

                label = prediction[5].replace(root_dir, "").split("/")
                label = label[2]
                if label in new_predictions:
                    new_predictions[label] += [[prediction[0], prediction[1], prediction[2], prediction[3]]]
                else:
                    new_predictions[label] = [[prediction[0], prediction[1], prediction[2], prediction[3]]]

            mean_predictions = {}
            for key in new_predictions.keys():
                # import pdb
                # pdb.set_trace()
                mean_predictions[key] = [sum(float(d[0]) for d in new_predictions[key]) / len(new_predictions[key]),
                                         sum(float(d[1]) for d in new_predictions[key]) / len(new_predictions[key]),
                                         sum(float(d[2]) for d in new_predictions[key]) / len(new_predictions[key]),
                                         sum(float(d[3]) for d in new_predictions[key]) / len(new_predictions[key])]

            # CHANGE HERE
            values_aux_af = []
            values_aux_as = []
            values_aux_in = []
            values_aux_ca = []
            total = 0
            for key, value in mean_predictions.items():
                f = lambda i: value[i]
                index = max(range(len(value)), key=f)
                # values_aux_ca.append(value[0])
                # values_aux_as.append(value[1])
                # values_aux_in.append(value[2])
                # values_aux_af.append(value[3])
                if index == 0:
                    caucasian_samples[key] = value[0]
                elif index == 1:
                    asian_samples[key] = value[1]
                elif index == 2:
                    indian_samples[key] = value[2]
                elif index == 3:
                    african_samples[key] = value[3]
                total += 1

            ###
            # print(np.sum(values_aux_ca))
            # print(np.sum(values_aux_as))
            # print(np.sum(values_aux_in))
            # print(np.sum(values_aux_af))
            # print(np.sum(values_aux_ca)/total)
            # print(np.sum(values_aux_as)/total)
            # print(np.sum(values_aux_in)/total)
            # print(np.sum(values_aux_af)/total)
            # print(total)
            ###

        samples_removed = []
        indians_removed = 0
        caucasians_removed = 0
        asians_removed = 0
        africans_removed = 0
        values_ca, keys_ca = (list(t) for t in
                              zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
        values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
        values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
        values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))

        fig, axs = plt.subplots(nrows=2, ncols=2, )
        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_af), bw=0.5, ax=axs[0][0], label=self.legend_dict[len(samples_removed)])
        plt.xlim(0, 1)

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_as), bw=0.5, ax=axs[0][1], label=self.legend_dict[len(samples_removed)])
        plt.xlim(0, 1)

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_in), bw=0.5, ax=axs[1][0], label=self.legend_dict[len(samples_removed)])
        plt.xlim(0, 1)

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_ca), bw=0.5, ax=axs[1][1], label=self.legend_dict[len(samples_removed)])
        plt.xlim(0, 1)
        while len(samples_removed) < self.to_sample:
            values_ca, keys_ca = (list(t) for t in
                                  zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
            values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
            values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
            values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))

            if len(samples_removed) in [1000,3500,7000]:
               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_af), bw=0.5, ax=axs[0][0], label=self.legend_dict[len(samples_removed)])

               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_as), bw=0.5, ax=axs[0][1], label=self.legend_dict[len(samples_removed)])

               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_in), bw=0.5, ax=axs[1][0], label=self.legend_dict[len(samples_removed)])

               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_ca), bw=0.5, ax=axs[1][1], label=self.legend_dict[len(samples_removed)])

            remove_step = 1
            if np.mean(values_ca) < np.mean(values_as) and np.mean(values_ca) < np.mean(values_af) and np.mean(
                    values_ca) < np.mean(values_in):
                # print("Removing Caucasian")
                caucasian_samples = remove_from_dict(keys_ca[:remove_step], caucasian_samples)
                samples_removed += keys_ca[:remove_step]
                caucasians_removed += remove_step
                continue

            if np.mean(values_as) < np.mean(values_ca) and np.mean(values_as) < np.mean(values_af) and np.mean(
                    values_as) < np.mean(values_in):
                # print("Removing Asian")
                asian_samples = remove_from_dict(keys_as[:remove_step], asian_samples)
                samples_removed += keys_as[:remove_step]
                asians_removed += remove_step
                continue

            if np.mean(values_in) < np.mean(values_ca) and np.mean(values_in) < np.mean(values_af) and np.mean(
                    values_in) < np.mean(values_as):
                # print("Removing Indian")
                indian_samples = remove_from_dict(keys_in[:remove_step], indian_samples)
                samples_removed += keys_in[:remove_step]
                indians_removed += remove_step
                continue

            # print("Removing African")
            african_samples = remove_from_dict(keys_af[:remove_step], african_samples)
            samples_removed += keys_af[:remove_step]
            africans_removed += remove_step
        values_ca, keys_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
        values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
        values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
        values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))
        print("Means")
        print(np.mean(values_ca))
        print(np.mean(values_as))
        print(np.mean(values_in))
        print(np.mean(values_af))
        print(caucasians_removed)
        print(asians_removed)
        print(indians_removed)
        print(africans_removed)
        print(len(values_ca))
        print(len(values_as))
        print(len(values_in))
        print(len(values_af))
        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_af), bw=0.5, ax=axs[0][0], label=self.legend_dict[len(samples_removed)])
        axs[0][0].set_xlim(0.5, 1)
        axs[0][0].set_title("African Distribution")
        axs[0][0].legend()

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_as), bw=0.5, ax=axs[0][1], label=self.legend_dict[len(samples_removed)])
        axs[0][1].set_xlim(0.5, 1)
        axs[0][1].set_title("Asian Distribution")
        axs[0][1].legend()

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_in), bw=0.5, ax=axs[1][0], label=self.legend_dict[len(samples_removed)])
        axs[1][0].set_xlim(0.5, 1)
        axs[1][0].set_title("Indian Distribution")
        axs[1][0].legend()

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_ca), bw=0.5, ax=axs[1][1], label=self.legend_dict[len(samples_removed)])
        axs[1][1].set_xlim(0.5, 1)
        axs[1][1].set_title("Caucasian Distribution")
        axs[1][1].legend()
        plt.tight_layout()
        plt.savefig('../ethnicity_score_plots/protocol_a_relabel' + str(self.to_sample) + '.png', dpi=300)
        plt.show()
        mean_africans = np.mean(values_af)
        mean_caucasians = np.mean(values_ca)
        mean_asians = np.mean(values_as)
        mean_indians = np.mean(values_in)
        for path in self.image_paths:
            if path.replace(root_dir, "").split("/")[2] in samples_removed:
                continue
            new_image_paths.append(path)
        self.image_paths = new_image_paths

    def protocol_d(self, new_image_paths, root_dir):
        african_samples = {}
        caucasian_samples = {}
        asian_samples = {}
        indian_samples = {}
        african_n_img = {}
        caucasian_n_img = {}
        asian_n_img = {}
        indian_n_img = {}
        for base in ["African", "Caucasian", "Asian", "Indian"]:
            print(base)
            with open('/Users/pedroneto/Desktop/ethnicity scores/predictions.txt') as f:
                predictions = f.readlines()

                predictions = [line.split(",") for line in predictions if base in line]
                # import pdb
                # pdb.set_trace()
                new_predictions = {}
                index = 0
                if base == "Caucasian": index = 0
                if base == "Asian": index = 1
                if base == "Indian": index = 2
                if base == "African": index = 3
                for prediction in predictions:
                    if "aligned_tmp" in root_dir:
                        prediction[5] = prediction[5].replace("aligned", "aligned_tmp")
                    if "aligned_tmp_tmp" in root_dir:
                        prediction[5] = prediction[5].replace("aligned", "aligned_tmp")

                    label = prediction[5].replace(root_dir, "").split("/")
                    label = label[2]
                    if label in new_predictions:
                        new_predictions[label].append(float(prediction[index]))
                    else:
                        new_predictions[label] = [float(prediction[index])]

                mean_predictions = {}
                num_images = {}
                for key in new_predictions.keys():
                    mean_predictions[key] = sum(d for d in new_predictions[key])  # / len(new_predictions[key])
                    num_images[key] = len(new_predictions[key])

                if base == "Caucasian":
                    caucasian_samples = mean_predictions.copy()
                    caucasian_n_img = num_images.copy()
                if base == "Asian":
                    asian_samples = mean_predictions.copy()
                    asian_n_img = num_images.copy()
                if base == "Indian":
                    indian_samples = mean_predictions.copy()
                    indian_n_img = num_images.copy()
                if base == "African":
                    african_samples = mean_predictions.copy()
                    african_n_img = num_images.copy()
        samples_removed = []
        indians_removed = 0
        caucasians_removed = 0
        asians_removed = 0
        africans_removed = 0
        values_ca, keys_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
        values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
        values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
        values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))
        _, num_images_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_n_img.values()))))
        _, num_images_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_n_img.values()))))
        _, num_images_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_n_img.values()))))
        _, num_images_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_n_img.values()))))
        print("N Images")
        print(np.sum(num_images_ca))
        print(np.sum(num_images_as))
        print(np.sum(num_images_in))
        print(np.sum(num_images_af))
        print("Means")
        print(np.mean(values_ca))
        print(np.mean(values_as))
        print(np.mean(values_in))
        print(np.mean(values_af))
        print("Sums")
        print(np.sum(values_ca))
        print(np.sum(values_as))
        print(np.sum(values_in))
        print(np.sum(values_af))

        fig, axs = plt.subplots(nrows=2, ncols=2, )
        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_af), bw=0.5, ax=axs[0][0], label=self.legend_dict[len(samples_removed)])
        #plt.xlim(0, 1)

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_as), bw=0.5, ax=axs[0][1], label=self.legend_dict[len(samples_removed)])
        #plt.xlim(0, 1)

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_in), bw=0.5, ax=axs[1][0], label=self.legend_dict[len(samples_removed)])
        #plt.xlim(0, 1)

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_ca), bw=0.5, ax=axs[1][1], label=self.legend_dict[len(samples_removed)])
        #plt.xlim(0, 1)

        while len(samples_removed) < self.to_sample:
            # print(len(values_ca))
            # print(len(values_as))
            # print(len(values_in))
            # print(len(values_af))
            values_ca, keys_ca = (list(t) for t in
                                  zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
            values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
            values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
            values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))
            if len(samples_removed) in [1000,3500,7000]:
               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_af), bw=0.5, ax=axs[0][0], label=self.legend_dict[len(samples_removed)])

               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_as), bw=0.5, ax=axs[0][1], label=self.legend_dict[len(samples_removed)])

               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_in), bw=0.5, ax=axs[1][0], label=self.legend_dict[len(samples_removed)])

               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_ca), bw=0.5, ax=axs[1][1], label=self.legend_dict[len(samples_removed)])

            remove_step = 1
            if np.mean(values_ca) < np.mean(values_as) and np.mean(values_ca) < np.mean(values_af) and np.mean(
                    values_ca) < np.mean(values_in):
                # print("Removing Caucasian")
                caucasian_samples = remove_from_dict(keys_ca[:remove_step], caucasian_samples)
                caucasian_n_img = remove_from_dict(keys_ca[:remove_step], caucasian_n_img)
                samples_removed += keys_ca[:remove_step]
                caucasians_removed += remove_step
                continue

            if np.mean(values_as) < np.mean(values_ca) and np.mean(values_as) < np.mean(values_af) and np.mean(
                    values_as) < np.mean(values_in):
                # print("Removing Asian")
                asian_samples = remove_from_dict(keys_as[:remove_step], asian_samples)
                asian_n_img = remove_from_dict(keys_as[:remove_step], asian_n_img)
                samples_removed += keys_as[:remove_step]
                asians_removed += remove_step
                continue

            if np.mean(values_in) < np.mean(values_ca) and np.mean(values_in) < np.mean(values_af) and np.mean(
                    values_in) < np.mean(values_as):
                # print("Removing Indian")
                indian_samples = remove_from_dict(keys_in[:remove_step], indian_samples)
                indian_n_img = remove_from_dict(keys_in[:remove_step], indian_n_img)
                samples_removed += keys_in[:remove_step]
                indians_removed += remove_step
                continue

            # print("Removing African")
            african_samples = remove_from_dict(keys_af[:remove_step], african_samples)
            african_n_img = remove_from_dict(keys_af[:remove_step], african_n_img)
            samples_removed += keys_af[:remove_step]
            africans_removed += remove_step
        values_ca, keys_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
        values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
        values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
        values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))
        _, num_images_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_n_img.values()))))
        _, num_images_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_n_img.values()))))
        _, num_images_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_n_img.values()))))
        _, num_images_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_n_img.values()))))
        print("N Images")
        print(np.sum(num_images_ca))
        print(np.sum(num_images_as))
        print(np.sum(num_images_in))
        print(np.sum(num_images_af))
        print("Means")
        print(np.mean(values_ca))
        print(np.mean(values_as))
        print(np.mean(values_in))
        print(np.mean(values_af))
        print("Sums")
        print(np.sum(values_ca))
        print(np.sum(values_as))
        print(np.sum(values_in))
        print(np.sum(values_af))
        print("Removed")
        print(caucasians_removed)
        print(asians_removed)
        print(indians_removed)
        print(africans_removed)
        print("Retained")
        print(len(values_ca))
        print(len(values_as))
        print(len(values_in))
        print(len(values_af))
        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_af), bw=0.5, ax=axs[0][0], label=self.legend_dict[len(samples_removed)])
        axs[0][0].set_xlim(0, 150)
        axs[0][0].set_title("African Distribution")
        axs[0][0].legend()

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_as), bw=0.5, ax=axs[0][1], label=self.legend_dict[len(samples_removed)])
        axs[0][1].set_xlim(0, 150)
        axs[0][1].set_title("Asian Distribution")
        axs[0][1].legend()

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_in), bw=0.5, ax=axs[1][0], label=self.legend_dict[len(samples_removed)])
        axs[1][0].set_xlim(0, 150)
        axs[1][0].set_title("Indian Distribution")
        axs[1][0].legend()

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_ca), bw=0.5, ax=axs[1][1], label=self.legend_dict[len(samples_removed)])
        axs[1][1].set_xlim(0, 150)
        axs[1][1].set_title("Caucasian Distribution")
        axs[1][1].legend()
        plt.tight_layout()
        plt.savefig('../ethnicity_score_plots/protocol_d' + str(self.to_sample) + '.png', dpi=300)
        plt.show()
        mean_africans = np.mean(values_af)
        mean_caucasians = np.mean(values_ca)
        mean_asians = np.mean(values_as)
        mean_indians = np.mean(values_in)
        for path in self.image_paths:
            if path.replace(root_dir, "").split("/")[2] in samples_removed:
                continue
            new_image_paths.append(path)
        self.image_paths = new_image_paths

    def protocol_e(self, new_image_paths, root_dir):
        african_samples = {}
        caucasian_samples = {}
        asian_samples = {}
        indian_samples = {}
        african_n_img = {}
        caucasian_n_img = {}
        asian_n_img = {}
        indian_n_img = {}
        for base in ["African", "Caucasian", "Asian", "Indian"]:
            print(base)
            with open('/Users/pedroneto/Desktop/ethnicity scores/predictions.txt') as f:
                predictions = f.readlines()

                predictions = [line.split(",") for line in predictions if base in line]
                # import pdb
                # pdb.set_trace()
                new_predictions = {}
                index = 0
                if base == "Caucasian": index = 0
                if base == "Asian": index = 1
                if base == "Indian": index = 2
                if base == "African": index = 3
                for prediction in predictions:
                    if "aligned_tmp" in root_dir:
                        prediction[5] = prediction[5].replace("aligned", "aligned_tmp")
                    if "aligned_tmp_tmp" in root_dir:
                        prediction[5] = prediction[5].replace("aligned", "aligned_tmp")

                    label = prediction[5].replace(root_dir, "").split("/")
                    label = label[2]
                    if label in new_predictions:
                        new_predictions[label].append(float(prediction[index]))
                    else:
                        new_predictions[label] = [float(prediction[index])]

                mean_predictions = {}
                num_images = {}
                for key in new_predictions.keys():
                    mean_predictions[key] = sum(d for d in new_predictions[key])  # / len(new_predictions[key])
                    num_images[key] = len(new_predictions[key])

                if base == "Caucasian":
                    caucasian_samples = mean_predictions.copy()
                    caucasian_n_img = num_images.copy()
                if base == "Asian":
                    asian_samples = mean_predictions.copy()
                    asian_n_img = num_images.copy()
                if base == "Indian":
                    indian_samples = mean_predictions.copy()
                    indian_n_img = num_images.copy()
                if base == "African":
                    african_samples = mean_predictions.copy()
                    african_n_img = num_images.copy()
        samples_removed = []
        indians_removed = 0
        caucasians_removed = 0
        asians_removed = 0
        africans_removed = 0
        values_ca, keys_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
        values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
        values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
        values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))
        _, num_images_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_n_img.values()))))
        _, num_images_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_n_img.values()))))
        _, num_images_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_n_img.values()))))
        _, num_images_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_n_img.values()))))
        print("N Images")
        print(np.sum(num_images_ca))
        print(np.sum(num_images_as))
        print(np.sum(num_images_in))
        print(np.sum(num_images_af))
        print("Means")
        print(np.mean(values_ca))
        print(np.mean(values_as))
        print(np.mean(values_in))
        print(np.mean(values_af))
        print("Sums")
        print(np.sum(values_ca))
        print(np.sum(values_as))
        print(np.sum(values_in))
        print(np.sum(values_af))

        fig, axs = plt.subplots(nrows=2, ncols=2, )
        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_af), bw=0.5, ax=axs[0][0], label=self.legend_dict[len(samples_removed)])
        # plt.xlim(0, 1)

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_as), bw=0.5, ax=axs[0][1], label=self.legend_dict[len(samples_removed)])
        # plt.xlim(0, 1)

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_in), bw=0.5, ax=axs[1][0], label=self.legend_dict[len(samples_removed)])
        # plt.xlim(0, 1)

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_ca), bw=0.5, ax=axs[1][1], label=self.legend_dict[len(samples_removed)])
        # plt.xlim(0, 1)
        while len(samples_removed) < self.to_sample:
            # print(len(values_ca))
            # print(len(values_as))
            # print(len(values_in))
            # print(len(values_af))
            values_ca, keys_ca = (list(t) for t in
                                  zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
            values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
            values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
            values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))

            if len(samples_removed) in [1000,3500,7000]:
               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_af), bw=0.5, ax=axs[0][0], label=self.legend_dict[len(samples_removed)])

               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_as), bw=0.5, ax=axs[0][1], label=self.legend_dict[len(samples_removed)])

               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_in), bw=0.5, ax=axs[1][0], label=self.legend_dict[len(samples_removed)])

               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_ca), bw=0.5, ax=axs[1][1], label=self.legend_dict[len(samples_removed)])

            remove_step = 1
            if np.sum(values_ca) > np.sum(values_as) and np.sum(values_ca) > np.sum(values_af) and np.sum(
                    values_ca) > np.sum(values_in):
                # print("Removing Caucasian")
                caucasian_samples = remove_from_dict(keys_ca[:remove_step], caucasian_samples)
                caucasian_n_img = remove_from_dict(keys_ca[:remove_step], caucasian_n_img)
                samples_removed += keys_ca[:remove_step]
                caucasians_removed += remove_step
                continue

            if np.sum(values_as) > np.sum(values_ca) and np.sum(values_as) > np.sum(values_af) and np.sum(
                    values_as) > np.sum(values_in):
                # print("Removing Asian")
                asian_samples = remove_from_dict(keys_as[:remove_step], asian_samples)
                asian_n_img = remove_from_dict(keys_as[:remove_step], asian_n_img)
                samples_removed += keys_as[:remove_step]
                asians_removed += remove_step
                continue

            if np.sum(values_in) > np.sum(values_ca) and np.sum(values_in) > np.sum(values_af) and np.sum(
                    values_in) > np.sum(values_as):
                # print("Removing Indian")
                indian_samples = remove_from_dict(keys_in[:remove_step], indian_samples)
                indian_n_img = remove_from_dict(keys_in[:remove_step], indian_n_img)
                samples_removed += keys_in[:remove_step]
                indians_removed += remove_step
                continue

            # print("Removing African")
            african_samples = remove_from_dict(keys_af[:remove_step], african_samples)
            african_n_img = remove_from_dict(keys_af[:remove_step], african_n_img)
            samples_removed += keys_af[:remove_step]
            africans_removed += remove_step
        values_ca, keys_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
        values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
        values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
        values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))
        _, num_images_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_n_img.values()))))
        _, num_images_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_n_img.values()))))
        _, num_images_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_n_img.values()))))
        _, num_images_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_n_img.values()))))
        print("N Images")
        print(np.sum(num_images_ca))
        print(np.sum(num_images_as))
        print(np.sum(num_images_in))
        print(np.sum(num_images_af))
        print("Means")
        print(np.mean(values_ca))
        print(np.mean(values_as))
        print(np.mean(values_in))
        print(np.mean(values_af))
        print("Sums")
        print(np.sum(values_ca))
        print(np.sum(values_as))
        print(np.sum(values_in))
        print(np.sum(values_af))
        print("Removed")
        print(caucasians_removed)
        print(asians_removed)
        print(indians_removed)
        print(africans_removed)
        print("Retained")
        print(len(values_ca))
        print(len(values_as))
        print(len(values_in))
        print(len(values_af))
        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_af), bw=0.5, ax=axs[0][0], label=self.legend_dict[len(samples_removed)])
        axs[0][0].set_xlim(0, 150)
        axs[0][0].set_title("African Distribution")
        axs[0][0].legend()

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_as), bw=0.5, ax=axs[0][1], label=self.legend_dict[len(samples_removed)])
        axs[0][1].set_xlim(0, 150)
        axs[0][1].set_title("Asian Distribution")
        axs[0][1].legend()

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_in), bw=0.5, ax=axs[1][0], label=self.legend_dict[len(samples_removed)])
        axs[1][0].set_xlim(0, 150)
        axs[1][0].set_title("Indian Distribution")
        axs[1][0].legend()

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_ca), bw=0.5, ax=axs[1][1], label=self.legend_dict[len(samples_removed)])
        axs[1][1].set_xlim(0, 150)
        axs[1][1].set_title("Caucasian Distribution")
        axs[1][1].legend()
        plt.tight_layout()
        plt.savefig('../ethnicity_score_plots/protocol_e' + str(self.to_sample) + '.png', dpi=300)
        plt.show()
        mean_africans = np.mean(values_af)
        mean_caucasians = np.mean(values_ca)
        mean_asians = np.mean(values_as)
        mean_indians = np.mean(values_in)
        for path in self.image_paths:
            if path.replace(root_dir, "").split("/")[2] in samples_removed:
                continue
            new_image_paths.append(path)
        self.image_paths = new_image_paths

    def protocol_d_relabel(self, new_image_paths, root_dir):
        african_samples = {}
        caucasian_samples = {}
        asian_samples = {}
        indian_samples = {}
        african_n_img = {}
        caucasian_n_img = {}
        asian_n_img = {}
        indian_n_img = {}
        with open('/Users/pedroneto/Desktop/ethnicity scores/predictions.txt') as f:
            predictions = f.readlines()
        predictions = [line.split(",") for line in predictions]
        # import pdb
        # pdb.set_trace()
        new_predictions = {}
        index = 0
        for prediction in predictions:

            label = prediction[5].replace(root_dir, "").split("/")
            label = label[2]
            if label in new_predictions:
                new_predictions[label] += [[prediction[0], prediction[1], prediction[2], prediction[3]]]
            else:
                new_predictions[label] = [[prediction[0], prediction[1], prediction[2], prediction[3]]]
        mean_predictions = {}
        num_images = {}
        for key in new_predictions.keys():
            # import pdb
            # pdb.set_trace()
            mean_predictions[key] = [sum(float(d[0]) for d in new_predictions[key]),  # / len(new_predictions[key]),
                                     sum(float(d[1]) for d in new_predictions[key]),  # / len(new_predictions[key]),
                                     sum(float(d[2]) for d in new_predictions[key]),  # / len(new_predictions[key]),
                                     sum(float(d[3]) for d in new_predictions[key]), ]  # / len(new_predictions[key])]
            num_images[key] = len(new_predictions[key])
        # CHANGE HERE
        values_aux_af = []
        values_aux_as = []
        values_aux_in = []
        values_aux_ca = []
        total = 0
        print(len(mean_predictions.items()))
        for key, value in mean_predictions.items():
            f = lambda i: value[i]
            index = max(range(len(value)), key=f)
            # values_aux_ca.append(value[0])
            # values_aux_as.append(value[1])
            # values_aux_in.append(value[2])
            # values_aux_af.append(value[3])
            if index == 0:
                caucasian_samples[key] = value[0]
                caucasian_n_img[key] = num_images[key]
            elif index == 1:
                asian_samples[key] = value[1]
                asian_n_img[key] = num_images[key]
            elif index == 2:
                indian_samples[key] = value[2]
                indian_n_img[key] = num_images[key]
            elif index == 3:
                african_samples[key] = value[3]
                african_n_img[key] = num_images[key]
            total += 1
        samples_removed = []
        indians_removed = 0
        caucasians_removed = 0
        asians_removed = 0
        africans_removed = 0
        values_ca, keys_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
        values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
        values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
        values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))
        _, num_images_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_n_img.values()))))
        _, num_images_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_n_img.values()))))
        _, num_images_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_n_img.values()))))
        _, num_images_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_n_img.values()))))
        print("N Images")
        print(np.sum(num_images_ca))
        print(np.sum(num_images_as))
        print(np.sum(num_images_in))
        print(np.sum(num_images_af))
        print("Means")
        print(np.mean(values_ca))
        print(np.mean(values_as))
        print(np.mean(values_in))
        print(np.mean(values_af))
        print("Sums")
        print(np.sum(values_ca))
        print(np.sum(values_as))
        print(np.sum(values_in))
        print(np.sum(values_af))

        fig, axs = plt.subplots(nrows=2, ncols=2, )
        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_af), bw=0.5, ax=axs[0][0], label=self.legend_dict[len(samples_removed)])
        # plt.xlim(0, 1)

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_as), bw=0.5, ax=axs[0][1], label=self.legend_dict[len(samples_removed)])
        # plt.xlim(0, 1)

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_in), bw=0.5, ax=axs[1][0], label=self.legend_dict[len(samples_removed)])
        # plt.xlim(0, 1)

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_ca), bw=0.5, ax=axs[1][1], label=self.legend_dict[len(samples_removed)])
        # plt.xlim(0, 1)
        while len(samples_removed) < self.to_sample:
            # print(len(values_ca))
            # print(len(values_as))
            # print(len(values_in))
            # print(len(values_af))
            values_ca, keys_ca = (list(t) for t in
                                  zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
            values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
            values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
            values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))
            if len(samples_removed) in [1000,3500,7000]:
               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_af), bw=0.5, ax=axs[0][0], label=self.legend_dict[len(samples_removed)])

               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_as), bw=0.5, ax=axs[0][1], label=self.legend_dict[len(samples_removed)])

               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_in), bw=0.5, ax=axs[1][0], label=self.legend_dict[len(samples_removed)])

               sns.set_style('whitegrid')
               sns.kdeplot(np.array(values_ca), bw=0.5, ax=axs[1][1], label=self.legend_dict[len(samples_removed)])

            remove_step = 1
            if np.mean(values_ca) < np.mean(values_as) and np.mean(values_ca) < np.mean(values_af) and np.mean(
                    values_ca) < np.mean(values_in):
                # print("Removing Caucasian")
                caucasian_samples = remove_from_dict(keys_ca[:remove_step], caucasian_samples)
                caucasian_n_img = remove_from_dict(keys_ca[:remove_step], caucasian_n_img)
                samples_removed += keys_ca[:remove_step]
                caucasians_removed += remove_step
                continue

            if np.mean(values_as) < np.mean(values_ca) and np.mean(values_as) < np.mean(values_af) and np.mean(
                    values_as) < np.mean(values_in):
                # print("Removing Asian")
                asian_samples = remove_from_dict(keys_as[:remove_step], asian_samples)
                asian_n_img = remove_from_dict(keys_as[:remove_step], asian_n_img)
                samples_removed += keys_as[:remove_step]
                asians_removed += remove_step
                continue

            if np.mean(values_in) < np.mean(values_ca) and np.mean(values_in) < np.mean(values_af) and np.mean(
                    values_in) < np.mean(values_as):
                # print("Removing Indian")
                indian_samples = remove_from_dict(keys_in[:remove_step], indian_samples)
                indian_n_img = remove_from_dict(keys_in[:remove_step], indian_n_img)
                samples_removed += keys_in[:remove_step]
                indians_removed += remove_step
                continue

            # print("Removing African")
            african_samples = remove_from_dict(keys_af[:remove_step], african_samples)
            african_n_img = remove_from_dict(keys_af[:remove_step], african_n_img)
            samples_removed += keys_af[:remove_step]
            africans_removed += remove_step
        values_ca, keys_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
        values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
        values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
        values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))
        _, num_images_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_n_img.values()))))
        _, num_images_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_n_img.values()))))
        _, num_images_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_n_img.values()))))
        _, num_images_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_n_img.values()))))
        print("N Images")
        print(np.sum(num_images_ca))
        print(np.sum(num_images_as))
        print(np.sum(num_images_in))
        print(np.sum(num_images_af))
        print("Means")
        print(np.mean(values_ca))
        print(np.mean(values_as))
        print(np.mean(values_in))
        print(np.mean(values_af))
        print("Sums")
        print(np.sum(values_ca))
        print(np.sum(values_as))
        print(np.sum(values_in))
        print(np.sum(values_af))
        print("Removed")
        print(caucasians_removed)
        print(asians_removed)
        print(indians_removed)
        print(africans_removed)
        print("Retained")
        print(len(values_ca))
        print(len(values_as))
        print(len(values_in))
        print(len(values_af))

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_af), bw=0.5, ax=axs[0][0], label=self.legend_dict[len(samples_removed)])
        axs[0][0].set_xlim(0, 150)
        axs[0][0].set_title("African Distribution")
        axs[0][0].legend()

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_as), bw=0.5, ax=axs[0][1], label=self.legend_dict[len(samples_removed)])
        axs[0][1].set_xlim(0, 150)
        axs[0][1].set_title("Asian Distribution")
        axs[0][1].legend()

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_in), bw=0.5, ax=axs[1][0], label=self.legend_dict[len(samples_removed)])
        axs[1][0].set_xlim(0, 150)
        axs[1][0].set_title("Indian Distribution")
        axs[1][0].legend()

        sns.set_style('whitegrid')
        sns.kdeplot(np.array(values_ca), bw=0.5, ax=axs[1][1], label=self.legend_dict[len(samples_removed)])
        axs[1][1].set_xlim(0, 150)
        axs[1][1].set_title("Caucasian Distribution")
        axs[1][1].legend()
        plt.tight_layout()
        plt.savefig('../ethnicity_score_plots/protocol_d_relabel' + str(self.to_sample) + '.png', dpi=300)
        plt.show()
        mean_africans = np.mean(values_af)
        mean_caucasians = np.mean(values_ca)
        mean_asians = np.mean(values_as)
        mean_indians = np.mean(values_in)
        for path in self.image_paths:
            if path.replace(root_dir, "").split("/")[2] in samples_removed:
                continue
            new_image_paths.append(path)
        self.image_paths = new_image_paths

    def protocol_e_relabel(self, new_image_paths, root_dir, to_sample):
        african_samples = {}
        caucasian_samples = {}
        asian_samples = {}
        indian_samples = {}
        african_n_img = {}
        caucasian_n_img = {}
        asian_n_img = {}
        indian_n_img = {}
        with open('/Users/pedroneto/Desktop/ethnicity scores/predictions.txt') as f:
            predictions = f.readlines()
        predictions = [line.split(",") for line in predictions]
        # import pdb
        # pdb.set_trace()
        new_predictions = {}
        index = 0
        for prediction in predictions:

            label = prediction[5].replace(root_dir, "").split("/")
            label = label[2]
            if label in new_predictions:
                new_predictions[label] += [[prediction[0], prediction[1], prediction[2], prediction[3]]]
            else:
                new_predictions[label] = [[prediction[0], prediction[1], prediction[2], prediction[3]]]
        mean_predictions = {}
        num_images = {}
        for key in new_predictions.keys():
            # import pdb
            # pdb.set_trace()
            mean_predictions[key] = [sum(float(d[0]) for d in new_predictions[key]),  # / len(new_predictions[key]),
                                     sum(float(d[1]) for d in new_predictions[key]),  # / len(new_predictions[key]),
                                     sum(float(d[2]) for d in new_predictions[key]),  # / len(new_predictions[key]),
                                     sum(float(d[3]) for d in new_predictions[key]), ]  # / len(new_predictions[key])]
            num_images[key] = len(new_predictions[key])
        # CHANGE HERE
        values_aux_af = []
        values_aux_as = []
        values_aux_in = []
        values_aux_ca = []
        total = 0
        print(len(mean_predictions.items()))
        for key, value in mean_predictions.items():
            f = lambda i: value[i]
            index = max(range(len(value)), key=f)
            # values_aux_ca.append(value[0])
            # values_aux_as.append(value[1])
            # values_aux_in.append(value[2])
            # values_aux_af.append(value[3])
            if index == 0:
                caucasian_samples[key] = value[0]
                caucasian_n_img[key] = num_images[key]
            elif index == 1:
                asian_samples[key] = value[1]
                asian_n_img[key] = num_images[key]
            elif index == 2:
                indian_samples[key] = value[2]
                indian_n_img[key] = num_images[key]
            elif index == 3:
                african_samples[key] = value[3]
                african_n_img[key] = num_images[key]
            total += 1
        samples_removed = []
        indians_removed = 0
        caucasians_removed = 0
        asians_removed = 0
        africans_removed = 0
        values_ca, keys_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
        values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
        values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
        values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))
        _, num_images_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_n_img.values()))))
        _, num_images_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_n_img.values()))))
        _, num_images_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_n_img.values()))))
        _, num_images_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_n_img.values()))))
        print("N Images")
        print(np.sum(num_images_ca))
        print(np.sum(num_images_as))
        print(np.sum(num_images_in))
        print(np.sum(num_images_af))
        print("Means")
        print(np.mean(values_ca))
        print(np.mean(values_as))
        print(np.mean(values_in))
        print(np.mean(values_af))
        print("Sums")
        print(np.sum(values_ca))
        print(np.sum(values_as))
        print(np.sum(values_in))
        print(np.sum(values_af))
        while len(samples_removed) < to_sample:
            # print(len(values_ca))
            # print(len(values_as))
            # print(len(values_in))
            # print(len(values_af))
            values_ca, keys_ca = (list(t) for t in
                                  zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
            values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
            values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
            values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))

            remove_step = 1
            if np.sum(values_ca) > np.sum(values_as) and np.sum(values_ca) > np.sum(values_af) and np.sum(
                    values_ca) > np.sum(values_in):
                # print("Removing Caucasian")
                caucasian_samples = remove_from_dict(keys_ca[:remove_step], caucasian_samples)
                caucasian_n_img = remove_from_dict(keys_ca[:remove_step], caucasian_n_img)
                samples_removed += keys_ca[:remove_step]
                caucasians_removed += remove_step
                continue

            if np.sum(values_as) > np.sum(values_ca) and np.sum(values_as) > np.sum(values_af) and np.sum(
                    values_as) > np.sum(values_in):
                # print("Removing Asian")
                asian_samples = remove_from_dict(keys_as[:remove_step], asian_samples)
                asian_n_img = remove_from_dict(keys_as[:remove_step], asian_n_img)
                samples_removed += keys_as[:remove_step]
                asians_removed += remove_step
                continue

            if np.sum(values_in) > np.sum(values_ca) and np.sum(values_in) > np.sum(values_af) and np.sum(
                    values_in) > np.sum(values_as):
                # print("Removing Indian")
                indian_samples = remove_from_dict(keys_in[:remove_step], indian_samples)
                indian_n_img = remove_from_dict(keys_in[:remove_step], indian_n_img)
                samples_removed += keys_in[:remove_step]
                indians_removed += remove_step
                continue

            # print("Removing African")
            african_samples = remove_from_dict(keys_af[:remove_step], african_samples)
            african_n_img = remove_from_dict(keys_af[:remove_step], african_n_img)
            samples_removed += keys_af[:remove_step]
            africans_removed += remove_step
        values_ca, keys_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
        values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
        values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
        values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))
        _, num_images_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_n_img.values()))))
        _, num_images_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_n_img.values()))))
        _, num_images_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_n_img.values()))))
        _, num_images_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_n_img.values()))))
        print("N Images")
        print(np.sum(num_images_ca))
        print(np.sum(num_images_as))
        print(np.sum(num_images_in))
        print(np.sum(num_images_af))
        print("Means")
        print(np.mean(values_ca))
        print(np.mean(values_as))
        print(np.mean(values_in))
        print(np.mean(values_af))
        print("Sums")
        print(np.sum(values_ca))
        print(np.sum(values_as))
        print(np.sum(values_in))
        print(np.sum(values_af))
        print("Removed")
        print(caucasians_removed)
        print(asians_removed)
        print(indians_removed)
        print(africans_removed)
        print("Retained")
        print(len(values_ca))
        print(len(values_as))
        print(len(values_in))
        print(len(values_af))
        mean_africans = np.mean(values_af)
        mean_caucasians = np.mean(values_ca)
        mean_asians = np.mean(values_as)
        mean_indians = np.mean(values_in)
        for path in self.image_paths:
            if path.replace(root_dir, "").split("/")[2] in samples_removed:
                continue
            new_image_paths.append(path)
        self.image_paths = new_image_paths

    def protocol_multiple_forces(self, new_image_paths, root_dir, to_sample):
        caucasian_samples = {}
        african_samples = {}
        asian_samples = {}
        indian_samples = {}
        with open('/Users/pedroneto/Desktop/ethnicity scores/predictions.txt') as f:
            predictions = f.readlines()

            predictions = [line.replace(" ", "").split(",") for line in predictions]
            new_predictions = {}
            # import pdb
            # pdb.set_trace()
            for prediction in predictions:
                if "aligned_tmp" in root_dir:
                    prediction[5] = prediction[5].replace("aligned", "aligned_tmp")
                label = prediction[5].replace(root_dir, "").split("/")

                label = label[2]
                if label in new_predictions:
                    new_predictions[label] += [[prediction[0], prediction[1], prediction[2], prediction[3]]]
                else:
                    new_predictions[label] = [[prediction[0], prediction[1], prediction[2], prediction[3]]]

            mean_predictions = {}
            for key in new_predictions.keys():
                # import pdb
                # pdb.set_trace()
                mean_predictions[key] = [sum(float(d[0]) for d in new_predictions[key]),  # / len(new_predictions[key]),
                                         sum(float(d[1]) for d in new_predictions[key]),  # / len(new_predictions[key]),
                                         sum(float(d[2]) for d in new_predictions[key]),  # / len(new_predictions[key]),
                                         sum(float(d[3]) for d in new_predictions[key])]  # / len(new_predictions[key])]

            # CHANGE HERE
            values_aux_af = []
            values_aux_as = []
            values_aux_in = []
            values_aux_ca = []
            total = 0
            all_samples = {}
            for key, value in mean_predictions.items():
                f = lambda i: value[i]
                index = max(range(len(value)), key=f)
                values_aux_ca.append(value[0])
                values_aux_as.append(value[1])
                values_aux_in.append(value[2])
                values_aux_af.append(value[3])
                if index == 0:
                    caucasian_samples[key] = value[0]
                elif index == 1:
                    asian_samples[key] = value[1]
                elif index == 2:
                    indian_samples[key] = value[2]
                elif index == 3:
                    african_samples[key] = value[3]
                all_samples[key] = value
                total += 1

            ###
            print(np.sum(values_aux_ca))
            print(np.sum(values_aux_as))
            print(np.sum(values_aux_in))
            print(np.sum(values_aux_af))
            print(np.sum(values_aux_ca) / total)
            print(np.sum(values_aux_as) / total)
            print(np.sum(values_aux_in) / total)
            print(np.sum(values_aux_af) / total)
            # print(total)
            ###
            values_aux_ca = np.sum(values_aux_ca)
            values_aux_as = np.sum(values_aux_as)
            values_aux_in = np.sum(values_aux_in)
            values_aux_af = np.sum(values_aux_af)
        samples_removed = []
        indians_removed = 0
        caucasians_removed = 0
        asians_removed = 0
        africans_removed = 0
        while len(samples_removed) < to_sample:  # 10721:
            values_ca, keys_ca = (list(t) for t in
                                  zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
            values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
            values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
            values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))

            remove_step = 1
            if values_aux_ca > values_aux_as and values_aux_ca > values_aux_af and values_aux_ca > values_aux_in:
                # print("Removing Caucasian")
                caucasian_samples = remove_from_dict(keys_ca[:remove_step], caucasian_samples)
                key = keys_ca[:remove_step][0]
                samples_removed += keys_ca[:remove_step]
                caucasians_removed += remove_step
                values_aux_ca -= all_samples[key][0]
                values_aux_as -= all_samples[key][1]
                values_aux_in -= all_samples[key][2]
                values_aux_af -= all_samples[key][3]
                continue

            if values_aux_as > values_aux_ca and values_aux_as > values_aux_af and values_aux_as > values_aux_in:
                # print("Removing Asian")
                asian_samples = remove_from_dict(keys_as[:remove_step], asian_samples)
                key = keys_as[:remove_step][0]
                samples_removed += keys_as[:remove_step]
                asians_removed += remove_step
                values_aux_ca -= all_samples[key][0]
                values_aux_as -= all_samples[key][1]
                values_aux_in -= all_samples[key][2]
                values_aux_af -= all_samples[key][3]
                continue

            if values_aux_in > values_aux_ca and values_aux_in > values_aux_af and values_aux_in > values_aux_as:
                # print("Removing Indian")
                indian_samples = remove_from_dict(keys_in[:remove_step], indian_samples)
                key = keys_in[:remove_step][0]
                samples_removed += keys_in[:remove_step]
                indians_removed += remove_step
                values_aux_ca -= all_samples[key][0]
                values_aux_as -= all_samples[key][1]
                values_aux_in -= all_samples[key][2]
                values_aux_af -= all_samples[key][3]
                continue

            # print("Removing African")
            african_samples = remove_from_dict(keys_af[:remove_step], african_samples)
            key = keys_af[:remove_step][0]
            samples_removed += keys_af[:remove_step]
            africans_removed += remove_step
            values_aux_ca -= all_samples[key][0]
            values_aux_as -= all_samples[key][1]
            values_aux_in -= all_samples[key][2]
            values_aux_af -= all_samples[key][3]
        values_ca, keys_ca = (list(t) for t in zip(*sorted(zip(caucasian_samples.values(), caucasian_samples.keys()))))
        values_af, keys_af = (list(t) for t in zip(*sorted(zip(african_samples.values(), african_samples.keys()))))
        values_in, keys_in = (list(t) for t in zip(*sorted(zip(indian_samples.values(), indian_samples.keys()))))
        values_as, keys_as = (list(t) for t in zip(*sorted(zip(asian_samples.values(), asian_samples.keys()))))
        print("Finished removal")
        print(np.sum(values_aux_ca))
        print(np.sum(values_aux_as))
        print(np.sum(values_aux_in))
        print(np.sum(values_aux_af))
        print("Means")
        print(np.mean(values_ca))
        print(np.mean(values_as))
        print(np.mean(values_in))
        print(np.mean(values_af))
        print("Sums")
        print(np.sum(values_ca))
        print(np.sum(values_as))
        print(np.sum(values_in))
        print(np.sum(values_af))
        print("Removed")
        print(caucasians_removed)
        print(asians_removed)
        print(indians_removed)
        print(africans_removed)
        print("Retained")
        print(len(values_ca))
        print(len(values_as))
        print(len(values_in))
        print(len(values_af))
        mean_africans = np.mean(values_af)
        mean_caucasians = np.mean(values_ca)
        mean_asians = np.mean(values_as)
        mean_indians = np.mean(values_in)
        for path in self.image_paths:
            if path.replace(root_dir, "").split("/")[2] in samples_removed:
                continue
            new_image_paths.append(path)
        self.image_paths = new_image_paths
        return african_samples, asian_samples, caucasian_samples, indian_samples, mean_africans, mean_asians, mean_caucasians, mean_indians

    def protocol_random(self, root_dir, to_sample):
        labels_ = []
        for race in ["African", "Caucasian", "Asian", "Indian"]:
            labels = []
            print("Starting " + race)
            for path in self.image_paths:
                if path.replace(root_dir, "").split("/")[1] == race:
                    labels.append(path.replace(root_dir, "").split("/")[2])
            labels_ += list(set(labels))
        samples_to_remove = np.random.choice(labels_, to_sample, replace=False)
        new_image_paths = []
        for path in self.image_paths:
            if path.replace(root_dir, "").split("/")[2] in samples_to_remove:
                continue
            new_image_paths.append(path)
        print("Ending " + race)
        self.image_paths = new_image_paths
        labels_ = []
        for race in ["African", "Caucasian", "Asian", "Indian"]:
            labels = []
            print("Starting " + race)
            for path in self.image_paths:
                if path.replace(root_dir, "").split("/")[1] == race:
                    labels.append(race + "/" + path.replace(root_dir, "").split("/")[2])
            print(len(set(labels)))
            labels_ += list(set(labels))
        print(len(labels_))

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        image_id = image_path.replace(self.root_dir, "")
        label = self.labels[image_id.split("/")[2]]
        prob = self.probs[image_id.split("/")[2]]
        # mx.image.imdecode(img).asnumpy()
        # idx = self.imgidx[index]
        # s = self.imgrec.read_idx(idx)
        # header, img = mx.recordio.unpack(s)
        # label = header.label
        # if not isinstance(label, numbers.Number):
        #    label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.imread(image_path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, prob

    def __len__(self):
        return len(self.image_paths)

def remove_from_dict(keys, dict):
    for key in keys:
        del dict[key]
    return dict


dataset = MXFaceDataset("/nas-ctm01/datasets/public/BIOMETRICS/race_per_7000_aligned",0)