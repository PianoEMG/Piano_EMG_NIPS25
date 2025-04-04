import os
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import pickle as pkl
from tqdm import tqdm
import argparse
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import wandb
import matplotlib.pyplot as plt
import json

from networks.transformer_lrf import TransformerV2
from networks.vq_lrf import VQEncoderV3
from dataloader.dataset import KeyEmgDataset, create_dataloader
from cfg import config
from cfg.options import Options

def log_and_plot_batch(emg_data, keystroke, performance_name):
    seq_len, emg_channels, keystroke_channels = emg_data.shape[0], emg_data.shape[1], keystroke.shape[1]
    # print(seq_len, emg_channels, keystroke_channels)

    # for batch_idx in range(batch_size):
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    # Plot Keystroke Data
    pressed_key_num = 0
    for channel in range(keystroke_channels):
        if torch.mean(keystroke[:, channel]) != -1:
            pressed_key_num = pressed_key_num + 1
            axs[0].plot(range(seq_len), keystroke[:, channel].cpu().detach().numpy(), label=f'Key {channel+1}')
    print(f"pressed_key_num: {pressed_key_num}")
    axs[0].set_title('Keystroke')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Height')
    axs[0].legend()

    # Plot Ground Truth EMG Data
    for channel in range(emg_channels):
        axs[1].plot(range(seq_len), emg_data[:, channel].cpu().detach().numpy(), label=f'Muscle {channel+1}')
    axs[1].set_title('GT EMG')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend()


    plt.tight_layout()
    # wandb.log({f"Epoch {epoch} Sample {batch_idx}": wandb.Image(fig)})
    plt.savefig(f"{performance_name}.png")
    plt.close(fig)


# data_dir = 'C:\\Users\\ruofa\\Desktop\\Piano_Dataset\\emg_data'
# files = os.listdir(data_dir)
# files = [f for f in files if os.path.isfile(os.path.join(data_dir, f))]
# file_len = 0
# for file in tqdm(files):
#     with open(f'C:/Users/ruofa/Desktop/Piano_Dataset/emg_data/{file}', 'rb') as f2:
#         keystroke_data = pkl.load(f2)
#         keystroke_data = torch.tensor(keystroke_data).float()
#         file_len = file_len + keystroke_data.shape[0]
# print(file_len)

# performance_name = "t4_p19_1"
# with open(f'C:/Users/ruofa/Desktop/Piano_Dataset/keystroke_data/'+performance_name+'.pkl', 'rb') as f2:
#     keystroke_data = pkl.load(f2)
#     keystroke_data = torch.tensor(keystroke_data).float()
# with open(f'C:/Users/ruofa/Desktop/Piano_Dataset/emg_data/'+performance_name+'.pkl', 'rb') as f:
#     emg_data = pkl.load(f)
#     emg_data = torch.tensor(emg_data).float()
# print(keystroke_data.shape, emg_data.shape)
# log_and_plot_batch(emg_data, keystroke_data, performance_name)


def get_configurations():
    parser = Options(eval=False)
    opt = parser.get_options()
    opt.mode = "train"
    opt.precision = "bf16"
    opt.epoch = 2
    opt.batch_size = 8
    opt.batch_size_val = 32
    opt.num_threads = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_num = torch.cuda.device_count()
    parser.print_options(opt)
    return opt, device, device_num

# opt, device, device_num = get_configurations()
# val_dataset = KeyEmgDataset(mode="val", win_len=4, overlap_len=0)
# val_loader = create_dataloader(val_dataset, opt, "KeyEmgDataloader", device_num, False, opt.batch_size_val)
# print()
# loss_fn = nn.MSELoss()

# total_val_loss_zeros = 0
# for data in tqdm(val_loader):
#     out_zeros = torch.zeros(data['emg'].shape[0], data['emg'].shape[1], data['emg'].shape[2])
#     loss_zeros = loss_fn(data['emg'], out_zeros)
#     total_val_loss_zeros += loss_zeros.item()
# avg_val_loss_zeros = total_val_loss_zeros / len(val_loader)

# print(avg_val_loss_zeros)

temp_tensor = torch.randn(8, 1024, 6)
vq_encoder = VQEncoderV3(input_size=6, channels= [1024, 128], n_down=2)
out_tensor = vq_encoder(temp_tensor)
print(out_tensor.shape)
