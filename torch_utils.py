import numpy as np
from torchvision.datasets.folder import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torch
import time
import copy

# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model, criterion, optimizer, scheduler, dataloaders,
                num_epochs=25, model_name=None, early_stop=None):
    since = time.time()
    dataset_sizes = {x: len(dataloaders[x].sampler.indices) for x in dataloaders}
    print(f'dataset size: {dataset_sizes}')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 999.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    es_accum = 0
    result_df = pd.DataFrame(columns=['train_acc', 'train_loss', 'val_acc', 'val_loss'])
    for epoch in range(num_epochs):
	if es_accum > early_stop:
	    print(f'early stopped at epoch = {epoch}')
	    continue
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # print(inputs.shape)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
	
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
	    result_df.loc[epoch, phase+'_acc'] = epoch_acc
            result_df.loc[epoch, phase+'_loss'] = epoch_loss
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model, early_stopping
            if phase == 'val':
		if epoch_loss < best_loss:
                    best_acc = epoch_acc
		    best_loss = epoch_loss
                    es_accum = 0
		    best_model_wts = copy.deepcopy(model.state_dict())
		else:
		    es_accum += 1

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    if model_name is not None:
    	torch.save(model.state_dict(), model_name)
        print(f'model saved to {model_name}')
    return model, result_df


def split_train_val_loader(train_dataset, valid_dataset,
                           num_all_data, valid_size=0.1, batch_size=10, train_enlarge_factor=1,
                           pin_memory=True, num_workers=1, random_seed=1):

    np.random.seed(random_seed)
    indices = list(range(num_all_data))
    split = int(np.floor(valid_size * num_all_data))
    np.random.shuffle(indices)
    print(f'num of all data: {num_all_data}, validset size: {split}')
    train_idx, valid_idx = indices[split:], indices[:split]
    train_idx_large = np.repeat(train_idx, train_enlarge_factor)
    np.random.shuffle(train_idx_large)
    train_sampler = SubsetRandomSampler(train_idx_large)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader

def get_data_loader(path, data_transform, batch_size=1, num_workers=1):
   test_dataset = ImageFolder('./test/', data_transform)
   test_loader = DataLoader(
       test_dataset, batch_size=batch_size, sampler=None,
       num_workers=num_workers, pin_memory=True,
       )
   return test_loader



