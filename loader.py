import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import random
from transform import *
from sklearn.model_selection import KFold
import random


class EEGDataLoader(Dataset):
    def __init__(self, config, fold, set = 'train', shuffle = 'True'):
        random.seed(316)
        self.dset_cfg = config['dataset']
        self.dset_name = self.dset_cfg['name']

        if self.dset_name == "SleepEDF78":
            subject_idx = list(range(83))
            subject_idx.remove(68)
            subject_idx.remove(69)
            subject_idx.remove(39)
            subject_idx.remove(78)
            subject_idx.remove(79)

            kf = KFold(n_splits=10, shuffle=True, random_state=123456)
            cross_val_idx_list = []
            for train_idx, test_val_idx in kf.split(subject_idx):
                random.shuffle(train_idx)
                val_idx = train_idx[:7]
                fold_data = {
                    'val': val_idx.tolist(),
                    'test': test_val_idx.tolist()
                }
                cross_val_idx_list.append(fold_data)

        if self.dset_name == "MASS3":
            subject_idx = list(range(1,65))
            subject_idx.remove(43)
            subject_idx.remove(49)
            kf = KFold(n_splits=31, shuffle=True, random_state=42)
            cross_val_idx_list = []
            for train_idx, test_val_idx in kf.split(subject_idx):
                random.shuffle(train_idx)
                val_idx = train_idx[:15]
                fold_data = {
                    'val': val_idx.tolist(),
                    'test': test_val_idx.tolist()
                }
                cross_val_idx_list.append(fold_data)
        
        if self.dset_name == "SleepEDF20":
            subject_idx = list(range(20))

            kf = KFold(n_splits=20, shuffle=True, random_state=42)
            cross_val_idx_list = []
            for train_idx, test_val_idx in kf.split(subject_idx):
                random.shuffle(train_idx)
                val_idx = train_idx[:4]
                fold_data = {
                    'val': val_idx.tolist(),
                    'test': test_val_idx.tolist()
                }
                cross_val_idx_list.append(fold_data)
                
        #print(cross_val_idx_list)
        self.cross_val_idx_list = cross_val_idx_list
        self.set = set
        self.fold = fold
        
        self.sr = self.dset_cfg['samplingrate']   
        self.split_len = self.dset_cfg['split_len']   
        self.root_dir = self.dset_cfg['root_dir']
        self.num_splits = self.dset_cfg['num_splits']
        self.eeg_channel = self.dset_cfg['eeg_channel']
        
        self.seq_len = self.dset_cfg['seq_len']
        self.target_idx = self.dset_cfg['target_idx']
        
        self.training_mode = config['training_params']['mode']
        self.step = config['training_params']['step']

        self.seq_len_start = int((self.seq_len -1 )/2)
        self.seq_len_end = int((self.seq_len +1 )/2)


        self.dataset_path = os.path.join(self.root_dir, 'Data', self.dset_name)
        self.inputs, self.labels, self.subject_lables, self.epochs = self.split_dataset()

        if self.step == 'pretrain':
            self.transform = Compose(
                transforms=[
                    RandomAmplitudeScale(),
                    RandomTimeShift(),
                    RandomDCShift(),
                    RandomZeroMasking(),
                    RandomAdditiveGaussianNoise(),
                    RandomBandStopFilter(),
                ]
            )
            self.two_transform = TwoTransform(self.transform)
        
        # if shuffle:
        #     random.shuffle(self.epochs)

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        n_sample = 30 * self.sr * self.seq_len
        file_idx, idx, seq_len = self.epochs[idx]
        if self.step == 'pretrain':
            inputs = self.inputs[file_idx][idx + self.seq_len_start:idx + self.seq_len_end] # only one center epochs
            #inputs = self.inputs[file_idx][idx : idx + 1] # only one center epochs
            # inputs = self.inputs[file_idx][idx:idx+seq_len]
            if inputs.shape != (1,3000):
                inputs = self.inputs[file_idx][-1:] # only one center epochs

            # inputs = self.inputs[file_idx][idx:idx+seq_len]
            input_a, input_b = self.two_transform(inputs)
            input_a = torch.from_numpy(input_a).float()
            input_b = torch.from_numpy(input_b).float()
            # inputs = torch.from_numpy(inputs).float() # for 1 augmentation 

            inputs = [input_a, input_b]
            # inputs = [inputs, input_b] # for 1 augmentation 

            labels = self.labels[file_idx][idx:idx+seq_len]

        else:
            inputs = self.inputs[file_idx][idx:idx+seq_len]
            inputs = torch.from_numpy(inputs).float()
            labels = self.labels[file_idx][idx:idx+seq_len]
        subject_lables = self.subject_lables[file_idx]


        labels = torch.from_numpy(labels).long()

        if self.step != 'pretrain':
            labels = labels[self.target_idx]
            inputs = inputs.reshape(1, n_sample)

        subject_lables = torch.from_numpy(subject_lables).long()

        return inputs, labels, subject_lables


    def split_dataset(self):
        file_idx = 0
        inputs, labels,subject_lables , epochs = [], [], [], []
        data_root = os.path.join(self.dataset_path, self.eeg_channel)
        data_fname_list = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(data_root, '*.npz')))]
        data_fname_dict = {'train': [], 'test': [], 'val': []}
        
        
        
        if self.dset_name == 'SleepEDF78' or self.dset_name == 'SleepEDF20':
            for i in range(len(data_fname_list)):
                if int(data_fname_list[i][3:5]) in self.cross_val_idx_list[self.fold-1]['test']:
                    data_fname_dict['test'].append(data_fname_list[i])
                elif int(data_fname_list[i][3:5]) in self.cross_val_idx_list[self.fold-1]['val']:
                        data_fname_dict['val'].append(data_fname_list[i])
                else:
                    data_fname_dict['train'].append(data_fname_list[i])  
            print('Test subject number: ', self.cross_val_idx_list[self.fold-1]['test'])
            print('Val subject number: ', self.cross_val_idx_list[self.fold-1]['val'])
        

        if self.dset_name == 'MASS3':
            for i in range(len(data_fname_list)):
                if int(data_fname_list[i][8:10]) in self.cross_val_idx_list[self.fold-1]['test']:
                    data_fname_dict['test'].append(data_fname_list[i])
                elif int(data_fname_list[i][8:10]) in self.cross_val_idx_list[self.fold-1]['val']:
                        data_fname_dict['val'].append(data_fname_list[i])
                else:
                    data_fname_dict['train'].append(data_fname_list[i])  
            print('Test subject number: ', self.cross_val_idx_list[self.fold-1]['test'])
            print('Val subject number: ', self.cross_val_idx_list[self.fold-1]['val'])

        

        Temp_epochs = []
        if self.step == 'pretrain':
            epochs = np.zeros((1,self.split_len,3)).astype(int)

            for data_fname in data_fname_dict[self.set]:
                npz_file = np.load(os.path.join(data_root, data_fname))
                inputs.append(npz_file['x'])
                labels.append(npz_file['y'])
                subject_lables.append(npz_file['subject_N'])
                
                epoch_size = len(npz_file['x'])
                for i in range(len(npz_file['y']) - self.seq_len + 1):
                    Temp_epochs.append([file_idx, i, self.seq_len])
                random.shuffle(Temp_epochs)

                pad = len(Temp_epochs) % self.split_len
                if pad != 0:
                    # Calculate how many elements to duplicate
                    padding_needed = self.split_len - pad
                    # Duplicate the first padding_needed elements and append them
                    Temp_epochs.extend(Temp_epochs[:padding_needed])
                    
                Temp_epochs = np.array(Temp_epochs)
                Temp_epochs = Temp_epochs.reshape(-1,self.split_len,3)
                epochs = np.concatenate((epochs, Temp_epochs), axis=0)
                Temp_epochs = []
                file_idx += 1

            epochs = epochs[1:,:,:]
            np.random.shuffle(epochs)
            epochs = epochs.reshape(epochs.shape[0],self.split_len*3)
            epochs = epochs.reshape(-1,3)


        else:
        # if self.step == 'pretrain' or self.step == 'train' :
            for data_fname in data_fname_dict[self.set]:
                npz_file = np.load(os.path.join(data_root, data_fname))
                inputs.append(npz_file['x'])


                labels.append(npz_file['y'])
                subject_lables.append(npz_file['subject_N'])

                # epoch_size = len(npz_file['x']) // self.seq_len
                # for i in range(epoch_size):
                #     epochs.append([file_idx, i, self.seq_len])
                # file_idx += 1
                for i in range(len(npz_file['y']) - self.seq_len + 1):
                    epochs.append([file_idx, i, self.seq_len])
                file_idx += 1
      

        return inputs, labels, subject_lables, epochs
    
