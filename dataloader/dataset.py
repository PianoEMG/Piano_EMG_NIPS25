import torch
import torch.utils.data as data
import numpy as np
import os
import pickle as pkl
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import json

''' Keystroke and ground truth EMG dataset for training and testing'''
class KeyEmgDataset(data.Dataset):
    def __init__(self, mode, win_len, overlap_len):
        self.mode = mode
        self.win_len, self.overlap_len = win_len, overlap_len
        self.data = []
        with open(f'./preprocess/temp_p1.json', 'r') as f:
        # with open(f'./preprocess/temp_p1.json', 'r') as f:
            dataset_all_list = json.load(f)
        self.dataset_list = dataset_all_list[self.mode]
            
        print(f"Total {len(self.dataset_list)} performances in {self.mode} mode, loading {self.mode} dataset...")
        # print(f'Cropping data into windows length={self.win_len}')
        
        for file in tqdm(self.dataset_list):
            with open(f'C:/Users/ruofa/Desktop/Piano_Dataset/keystroke_data/{file}', 'rb') as f2:
                keystroke_data = pkl.load(f2)
            keystroke_data = torch.tensor(keystroke_data).float()
            with open(f'C:/Users/ruofa/Desktop/Piano_Dataset/emg_data/{file}', 'rb') as f:
                emg_data = pkl.load(f)
            emg_data = torch.tensor(emg_data).float()
            # print(keystroke_data.shape,emg_data.shape)
            if emg_data.shape[0] != keystroke_data.shape[0]:
                print(f"Error! {file}'s EMG frames number is not equal to keystroke")

            num_windows = (keystroke_data.shape[0] - self.overlap_len) // (self.win_len - self.overlap_len)
            split_keystroke, num_windows_with_last = self.segment_sliding_subarray(keystroke_data, self.win_len, self.overlap_len, num_windows, self.mode)
            split_emg, num_windows_with_last = self.segment_sliding_subarray(emg_data, self.win_len, self.overlap_len, num_windows, self.mode)
            for i in range(num_windows_with_last):
                self.data.append((split_keystroke[i], split_emg[i]))
    
    def segment_sliding_subarray(self, arr, win_len, overlap_len, num_windows, mode):
        sub_arrs = []
        arr = arr.numpy()
        if mode == "test" or mode == "val":
            overlap_len = 0
        step_len = win_len - overlap_len
        num_windows_with_last = num_windows
        for i in range(num_windows):
            start_idx = i * step_len
            end_idx = start_idx + win_len
            sub_arrs.append(arr[start_idx:end_idx])
        if len(arr) > end_idx: # if the last window is shorter than win_len
            num_windows_with_last = num_windows +1
            last_segment = arr[num_windows * step_len:]
            padded_segment = np.full((win_len, last_segment.shape[1]), fill_value=-1, dtype=arr.dtype)  # pad with -1 to a win_len length window
            padded_segment[:len(last_segment)] = last_segment
            sub_arrs.append(padded_segment)
        return np.array(sub_arrs), num_windows_with_last

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx] # is a tuple, like ([keystroke],[emg])
        return {
            "keystroke": data[0],
            "emg": data[1],
        }


def create_dataloader(dataset, args, dataset_type, device_num, shuffle, batch_size):
    DATALOADER = {
        'KeyEmgDataloader': KeyEmgDataset,
    }

    loader_cls = dataset_type
    assert loader_cls in DATALOADER.keys(), f'DataLoader {loader_cls} does not exist.'
    loader = DATALOADER[loader_cls]

    dataloader = data.DataLoader(
        dataset         = dataset,
        batch_size      = batch_size,
        shuffle         = shuffle,
        num_workers     = args.num_threads,
        # drop_last       = device_num > 1,
        drop_last = False,
        pin_memory      = False,
        prefetch_factor = 2 if args.num_threads > 0 else None,
        # collate_fn=collate_fn,
    )
    return dataloader
