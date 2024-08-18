import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import utils.transforms as T
import utils.utilities as utils
from tqdm import tqdm

class FWIDataset(Dataset):
    ''' FWI dataset
    For convenience, in this class, a batch refers to a npy file 
    instead of the batch used during training.

    Args:
        anno: path to annotation file
        preload: whether to load the whole dataset into memory
        sample_ratio: downsample ratio for seismic data
        file_size: # of samples in each npy file
        transform_data|label: transformation applied to data or label
    '''
    def __init__(
                    self, 
                    anno, 
                    preload=True, 
                    sample_ratio=1, 
                    file_size=500,
                    transform_data=None, 
                    transform_label=None, 
                    mask_factor=0.0
                ):
        if not os.path.exists(anno):
            print(f'Annotation file {anno} does not exists')
            
        self.preload = preload
        self.sample_ratio = sample_ratio
        self.file_size = file_size
        self.transform_data = transform_data
        self.transform_label = transform_label
        with open(anno, 'r') as f:
            self.batches = f.readlines()

        """
        THIS FUNCTION ONLY WORKS WITH PRELOAD. MASK_RATIO not implemented for preload=False
        """
        if preload:
            self.data_list, self.label_list= (), ()
            for batch in tqdm(self.batches):
                data, label = self.load_every(batch) 

                self.data_list = self.data_list + (data,)
                self.label_list = self.label_list + (label,)

            self.data_list = np.concatenate(self.data_list, 0)
            self.label_list = np.concatenate(self.label_list, 0)

            mask_indices = np.random.choice(len(self.data_list), 
                                              int(mask_factor*len(self.data_list)),
                                              replace=False)
            
            self.mask_list = np.ones(len(self.data_list), dtype=np.int8)
            self.mask_list[mask_indices] = 0
            
            print("Data concatenation complete.")
            if self.transform_label is not None:
                self.label_list = self.transform_label(self.label_list)
            if self.transform_data is not None:
                self.data_list = self.transform_data(self.data_list)

    # Load from one line
    def load_every(self, batch):
        batch = batch.split('\t')
        data_path = batch[0] if len(batch) > 1 else batch[0][:-1]
        data = np.load(data_path)[:, :, ::self.sample_ratio, :]
        data = data.astype('float32')
        if len(batch) > 1:
            label_path = batch[1][:-1]    
            label = np.load(label_path)
            label = label.astype('float32')
        else:
            label = None
        return data, label
    
    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
        if self.preload:
            mask = self.mask_list[idx]
            data = self.data_list[idx]
            label = self.label_list[idx] if len(self.label_list) != 0 else None
        else:
            data, label = self.load_every(self.batches[batch_idx])
            data = data[sample_idx]
            label = label[sample_idx] if label is not None else None
            if self.transform_data is not None:
                data = self.transform_data(data)
            if self.transform_label is not None:
                label = self.transform_data(label)
            mask=None #NOT-IMPLEMENTED
        return mask, data, label if label is not None else np.array([])
        
    def __len__(self):
        return len(self.batches) * self.file_size


if __name__ == '__main__':
    transform_data = Compose([
        T.LogTransform(k=1),
        T.MinMaxNormalize(T.log_transform(-61, k=1), T.log_transform(120, k=1))
    ])
    transform_label = Compose([
        T.MinMaxNormalize(2000, 6000)
    ])
    dataset = FWIDataset(f'relevant_files/temp.txt', transform_data=transform_data, transform_label=transform_label, file_size=1)
    data, label = dataset[0]
    print(data.shape)
    print(label is None)
