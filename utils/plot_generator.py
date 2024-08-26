import numpy as np
import matplotlib.pyplot as plt

def fiq_es(path1,path2):
    root_dir = "/nas-ctm01/datasets/public/BIOMETRICS/race_per_7000_aligned"
    for ethnicity in ("Caucasian","Asian","Indian","African"):
        with open(path1) as f:
            predictions = f.readlines()
        with open(path2) as f:
            quality = f.readlines()
        
        quality = [line.split(" ") for line in quality if ethnicity in line]
        predictions = [line.split(",") for line in predictions if ethnicity in line]
        print(len(quality))
        print(len(predictions))
        ethnicities = {"Caucasian":0,"Asian":1,"Indian":2,"African":3}

        new_quality = []
        new_predictions = []
        for prediction, qual in zip(predictions, quality):
            if qual[0].rstrip().lstrip() != prediction[5].rstrip().lstrip():
                print("Not supposed to happen")
                print(qual[0])
                print(prediction[5])
                break 
            new_quality.append(float(qual[1]))
            new_predictions.append(float(prediction[ethnicities[ethnicity]]))

        sizes = np.random.uniform(15, 80, len(new_predictions))
        colors = np.random.uniform(15, 80, len(new_predictions))
        fig, ax = plt.subplots()

        ax.scatter(new_predictions, new_quality, s=sizes, c=colors, vmin=0, vmax=100)

        ax.set(xlim=(0, 1),
            ylim=(0, 3), yticks=np.arange(1, 3))

        plt.savefig(ethnicity+"_es_quality_rgb.jpg")
        
        


        #mean_predictions= {}
        #for key in new_predictions.keys():
        #    mean_predictions[key] = sum(d for d in new_predictions[key]) / len(new_predictions[key])
        
        #print(np.mean(list(mean_predictions.values())))
    print("done")

def fiq_es_per_id(path1,path2):
    root_dir = "/nas-ctm01/datasets/public/BIOMETRICS/race_per_7000_aligned"
    for ethnicity in ("Caucasian","Asian","Indian","African"):
        with open(path1) as f:
            predictions = f.readlines()
        with open(path2) as f:
            quality = f.readlines()
        
        quality = [line.split(" ") for line in quality if ethnicity in line]
        predictions = [line.split(",") for line in predictions if ethnicity in line]
        print(len(quality))
        print(len(predictions))
        ethnicities = {"Caucasian":0,"Asian":1,"Indian":2,"African":3}

        new_predictions = {}
        for prediction in predictions: 
            label = prediction[5].replace(root_dir,"").split("/")[2]
            if label in new_predictions:
                new_predictions[label].append(float(prediction[ethnicities[ethnicity]]))
            else:
                new_predictions[label] = [float(prediction[ethnicities[ethnicity]])]

        new_quality = {}
        for qual in quality: 
            label = qual[0].replace(root_dir,"").split("/")[2]
            if label in new_quality:
                new_quality[label].append(float(qual[1]))
            else:
                new_quality[label] = [float(qual[1])]

        mean_predictions= {}
        for key in new_predictions.keys():
            mean_predictions[key] = sum(d for d in new_predictions[key]) / len(new_predictions[key])
        mean_quality = {}
        for key in new_quality.keys():
            try: 
                mean_predictions[key]
                mean_quality[key] = sum(d for d in new_quality[key]) / len(new_quality[key])
            except:
                continue

        new_quality = []
        new_predictions = []
        #import pdb
        #pdb.set_trace()
        for key in mean_predictions.keys():
            try:
                new_quality.append(mean_quality[key])
            except:
                continue
            new_predictions.append(mean_predictions[key])

        

        sizes = np.random.uniform(15, 80, len(new_predictions))
        colors = np.random.uniform(15, 80, len(new_predictions))
        fig, ax = plt.subplots()

        #ax.scatter(new_predictions, new_quality, s=sizes, c=colors, vmin=0, vmax=100)
        ax.hist2d(new_predictions, new_quality, bins=(np.arange(0, 1, 0.1), np.arange(0, 3, 0.1)))
        ax.set(xlim=(0, 1),
            ylim=(0, 3), yticks=np.arange(1, 3))

        plt.savefig(ethnicity+"_plot_id_es_quality.jpg")
        #print(np.mean(list(mean_predictions.values())))
    print("done")

#fiq_es('/nas-ctm01/homes/pdcarneiro/SkinToneClassifier/predictions.txt',"/nas-ctm01/homes/pdcarneiro/CR-FIQA/quality_data/balanced/CRFIQAL2.txt")
fiq_es_per_id('/nas-ctm01/homes/pdcarneiro/SkinToneClassifier/predictions.txt',"/nas-ctm01/homes/pdcarneiro/CR-FIQA/quality_data/balanced/CRFIQAL2.txt")

import pdb 
pdb.set_trace()