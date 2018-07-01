
from torch_utils import (split_train_val_loader, get_data_loader, 
                         train_model, get_model, predict_to_ensemble)
from preprocess import normalize_05, normalize_torch, preprocess, preprocess_hflip, preprocess_with_augmentation
import torch
import models
import utils
from augmentation import five_crops, HorizontalFlip, make_transforms
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder

model_name = "densenet161"
model_class = models.densenet161_finetune
model_state_pth = './densenet161-070102.pth'
image_size = 224
normalize = normalize_05
nb_classes = 15
BATCH_SIZE=15

# res = predict_to_ensemble(model_name, model_class, model_state_pth, image_size, normalize, nb_classes)

res = predict_to_ensemble('resnet50', models.resnet50_finetune, './resnet50-070102.pth', image_size, normalize, nb_classes)
res = predict_to_ensemble('resnet152', models.resnet152_finetune, './resnet152-070102.pth', image_size, normalize, nb_classes)


