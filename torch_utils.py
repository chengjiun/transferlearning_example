import numpy as np
from torchvision.datasets.folder import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

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



