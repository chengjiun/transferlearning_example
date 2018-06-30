from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder, DatasetFolder
from torch.autograd import Variable

import models
from torch_utils import split_train_val_loader
import utils
from utils import RunningMean, use_gpu
from preprocess import preprocess, preprocess_with_augmentation, normalize_05, normalize_torch

import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

NB_CLASSES = 15
BATCH_SIZE = 20 
IMAGE_SIZE = 224
VALID_SIZE = 0.2
TRAIN_ENLARGE_FACTOR = 5
EPOCH = 50 
MODEL_FILE_NAME = 'resnet152-070102.pth'
PATIENCE_LIMIT = 2
RANDOM_SEED = 152
MODEL = models.resnet152_finetune
EARLY_STOP = 10

def get_model():
    print('[+] loading model... ', end='', flush=True)
    model = MODEL(NB_CLASSES)
    if use_gpu:
        model.cuda()
    print('done')
    return model


def train():
    train_dataset = ImageFolder('./data/train/', transform=preprocess_with_augmentation(normalize_torch, IMAGE_SIZE))
    valid_dataset = ImageFolder('./data/train/', transform=preprocess(normalize_torch, IMAGE_SIZE))
    training_data_loader, valid_data_loader = (split_train_val_loader(train_dataset, valid_dataset,
                           len(train_dataset), valid_size=VALID_SIZE, batch_size=BATCH_SIZE,
			   train_enlarge_factor=TRAIN_ENLARGE_FACTOR,
                           pin_memory=True, num_workers=1, random_seed=RANDOM_SEED
                           ))


    model = get_model()

    criterion = nn.CrossEntropyLoss().cuda()

    nb_learnable_params = sum(p.numel() for p in model.fresh_params())
    print(f'[+] nb learnable params {nb_learnable_params}')

    lx, px = utils.predict(model, valid_data_loader, prob=False)
    min_loss = criterion(Variable(px), Variable(lx)).item()
    _, preds = torch.max(px.data, dim=1)
    accuracy = torch.mean((preds != lx).float())
    print(f' original loss: {min_loss}, accuracy: {accuracy}')

    lr = 0.001
    patience = 0
    earlystop = 0
    optimizer = torch.optim.Adam(model.fresh_params(), lr=lr)
    torch.save(model.state_dict(), MODEL_FILE_NAME)
    for epoch in range(EPOCH):
        if epoch == 1:
            lr = 0.0005
            print(f'[+] set lr={lr}')
        if patience == PATIENCE_LIMIT:
            patience = 0
            model.load_state_dict(torch.load(MODEL_FILE_NAME))
            lr = lr / 10
            print(f'[+] set lr={lr}')
        if earlystop > EARLY_STOP:
            model.load_state_dict(torch.load(MODEL_FILE_NAME))
            print('EARLY STOPPED')
            continue
        if epoch > 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

        running_loss = RunningMean()
        running_score = RunningMean()

        model.train()
        pbar = tqdm(training_data_loader, total=len(training_data_loader))
        for inputs, labels in pbar:
            batch_size = inputs.size(0)

            inputs = Variable(inputs)
            labels = Variable(labels)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim=1)

            loss = criterion(outputs, labels)
            running_loss.update(loss.item(), 1)
            running_score.update(torch.sum(preds == labels.data).float(), batch_size)

            loss.backward()
            optimizer.step()

            pbar.set_description(f'{epoch}: {running_loss.value:.5f} {running_score.value:.3f}')
	
        model.eval()
        lx, px = utils.predict(model, valid_data_loader)
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.item()
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds == lx).float())
        print(f'[+] val loss: {log_loss:.5f} acc: {accuracy:.3f}')

        if (log_loss < min_loss):
            torch.save(model.state_dict(), MODEL_FILE_NAME)
            print(f'[+] val loss improved from {min_loss:.5f} to {log_loss:.5f}, accuracy={accuracy}. Saved!')
            min_loss = log_loss
            patience = 0
        else:
            patience += 1
            earlystop += 1

if __name__ == "__main__":
    train()
