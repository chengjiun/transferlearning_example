import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import os
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.autograd import Variable
import torch

label_dict = {
    'CALsuburb':9,
    'PARoffice': 7,
    'bedroom':12,
    'coast':10,
    'forest':4,
    'highway': 14,
    'industrial': 2,
    'insidecity': 3,
    'kitchen': 0,
    'livingroom': 5,
    'mountain':8,
    'opencountry':6,
    'store':11,
    'street':1,
    'tallbuilding':13,
}

base_dir = './'

def load_image(img_file, y, shape=50):
    x_train = np.array([]).reshape((0, shape*shape))
    y_train = np.array([]).reshape((0, y.shape[1]))
    
    new_ind = shuffle(range(len(img_file)))
    x = img_file[new_ind]
    y = np.take(y, new_ind, axis=0)
    name = np.take(img_file, new_ind, axis=0)
    for i, (xi, yi) in enumerate(zip(x,y)):
        img = cv2.imread(xi, 0)
        img = cv2.resize(img, (shape,shape))
        x_train = np.row_stack([x_train, img.flatten()])
        y_train = np.row_stack([y_train, yi])
        
    x_batch = x_train.copy()
    x_batch /= 255.
    x_batch = x_batch.reshape((i+1, shape, shape, 1))
    y_batch = y_train.copy()
    return x_batch, y_batch

def load_train_df():
    data_dir = f'{base_dir}/data/train'
    train_dir = os.listdir(data_dir)
    train_pair_sr = pd.Series()
    for d in train_dir:
        for f in os.listdir(f'{data_dir}/{d}'):
            train_pair_sr[f'{data_dir}/{d}/{f}'] = label_dict[d]
    train_y = pd.get_dummies(train_pair_sr, '').as_matrix()#將label做 one_hot encoding
    train_pair_df = train_pair_sr.reset_index()
    train_pair_df.rename(columns={'index':'img',0:'y'},inplace=True)
    for i,file in enumerate(train_pair_df['img']):
        img = cv2.imread(file, 0)
        train_pair_df.loc[i,'sizex'] = img.shape[0]
        train_pair_df.loc[i,'sizey'] = img.shape[1]
        train_pair_df.loc[i,'file'] = file
        train_pair_df.loc[i,'v_max'] = img.max()
        train_pair_df.loc[i,'v_min'] = img.min()
        train_pair_df.loc[i,'v_std'] = img.std()
        train_pair_df.loc[i,'v_mean'] = img.mean()
        train_pair_df.loc[i,'v_median'] = np.median(img)
    
    train_pair_df['square'] = train_pair_df['sizex'] == train_pair_df['sizey']
    train_pair_df['size_ratio'] = train_pair_df['sizex'] / train_pair_df['sizey']
    return train_pair_df

def load_test_df():
    test_dir = os.listdir(f'{base_dir}/data/testset/')
    test_df = pd.DataFrame(columns=['img', 'test_x', 
                                    'sizex', 'sizey',
                                    'v_max', 'v_min', 'v_std', 'v_mean','v_median'])
    for i,f in enumerate(test_dir, shape):    
        test_df.loc[i,'img'] = f
        img = cv2.imread(f'{base_dir}/testset/{f}', 0)
        img = cv2.resize(img, (shape,shape)) / 256
        test_df.loc[i,'test_x'] = img.reshape((1,shape,shape,1))
        test_df.loc[i,'sizex'] = img.shape[0]
        test_df.loc[i,'sizey'] = img.shape[1]
        test_df.loc[i,'v_max'] = img.max()
        test_df.loc[i,'v_min'] = img.min()
        test_df.loc[i,'v_std'] = img.std()
        test_df.loc[i,'v_mean'] = img.mean()
        test_df.loc[i,'v_median'] = np.median(img)
    
    test_df['square'] = test_df['sizex'] == test_df['sizey']
    test_df['size_ratio'] = test_df['sizex'] / test_df['sizey']
    return test_df

use_gpu = torch.cuda.is_available()

class RunningMean:
    def __init__(self, value=0, count=0):
        self.total_value = value
        self.count = count

    def update(self, value, count=1):
        self.total_value += value
        self.count += count

    @property
    def value(self):
        if self.count:
            return self.total_value / self.count
        else:
            return float("inf")

    def __str__(self):
        return str(self.value)

def predict(model, dataloader, prob=False):
    all_labels = []
    all_outputs = []
    model.eval()

    pbar = tqdm(dataloader, total=len(dataloader))
    for inputs, labels in pbar:
        all_labels.append(labels)

        inputs = Variable(inputs)
        if use_gpu:
            inputs = inputs.cuda()

        outputs = model(inputs)
        if not prob:
            all_outputs.append(outputs.data.cpu())
        else:
            _, preds = torch.max(outputs.data, dim=1)
            all_outputs.append(preds.data.cpu())
            
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    if use_gpu:
        all_labels = all_labels.cuda()
        all_outputs = all_outputs.cuda()

    return all_labels, all_outputs


def safe_stack_2array(acc, a):
    a = a.unsqueeze(-1)
    if acc is None:
        return a
    return torch.cat((acc, a), dim=acc.dim() - 1)

def predict_tta(model, dataloaders):
    prediction = None
    lx = None
    for dataloader in dataloaders:
        lx, px = predict(model, dataloader)
        prediction = safe_stack_2array(prediction, px)

    return lx, prediction
