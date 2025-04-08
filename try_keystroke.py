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
from os.path import join as pjoin

from networks.transformer_lrf_VQ import TransformerV2
from networks.EMGFormer_model import EMGFormer, Seq2SeqTransformer
from networks.vq_lrf import VQEncoderV3, TransformerT2MTrainer
from dataloader.dataset import KeyEmgDataset, create_dataloader
from cfg import config
from cfg.options import Options
from networks.vq_lrf import VQEncoderV3, VQDecoderV3, Quantizer, VQTokenizerTrainerV3
from train_T2MT_vq_transformer import get_configurations

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



opt, device, device_num = get_configurations()

# train_dataset = KeyEmgDataset(mode="train", win_len=opt.win_len, overlap_len=opt.overlap)
# val_dataset = KeyEmgDataset(mode="val", win_len=opt.win_len_val, overlap_len=opt.overlap_val)
# train_loader = create_dataloader(train_dataset, opt, "KeyEmgDataloader", device_num, True, opt.batch_size)
# print(f"train dataset len / batch_size = dataloader len: {len(train_dataset)} / {opt.batch_size} = {len(train_loader)}")
# val_loader = create_dataloader(val_dataset, opt, "KeyEmgDataloader", device_num, False, opt.batch_size_val)
# print(f"val dataset len / batch_size = dataloader len: {len(val_dataset)} / {opt.batch_size_val} = {len(val_loader)}")

temp_keystroke = torch.randn(8, 1024, 88)
# temp_target_tensor = torch.randn(8, 128, 256)
# transformer = TransformerV2()
transformer = Seq2SeqTransformer()

# trainer = TransformerT2MTrainer(opt, transformer)
# trainer.train(train_loader, val_loader, None)

out = transformer(temp_keystroke)
print(f"out shape: {out.shape}")


# checkpoint = torch.load('../../Piano_EMG_NIPS25_checkpoints/VQ_Model/2025-04-04_18-02-21 128x256/finest.tar',
#                             map_location='cuda')
# dim_vq_latent = 256
# en_channels = [128, 256, dim_vq_latent]
# de_channels = [dim_vq_latent, 256, 128, 6]
# vq_encoder = VQEncoderV3(input_size=6, channels=en_channels, n_down=3)
# vq_decoder = VQDecoderV3(input_size=dim_vq_latent, channels=de_channels, n_resblk=2, n_up=3)
# quantizer = Quantizer(1024, dim_vq_latent, 1)

# vq_encoder.load_state_dict(checkpoint['vq_encoder'])    
# vq_decoder.load_state_dict(checkpoint['vq_decoder'])
# quantizer.load_state_dict(checkpoint['quantizer'])

# temp_emg = torch.randn(8, 1024, 6)
# pre_latents = vq_encoder(temp_emg)
# print(f"encoder_out shape: {pre_latents.shape}") # [bs, 128, 256]
# embedding_loss, vq_latents, _, perplexity = quantizer(pre_latents)
# print(f"embedding_loss: {embedding_loss}") # []
# print(f"vq_latents shape: {vq_latents.shape}") # [bs, 128, 256]   
# print(f"perplexity: {perplexity}") # 71.68    

# out_tensor = transformer(temp_keystroke, vq_latents)
# print(f"out_tensor shape: {out_tensor.shape}") # [bs, 128, 256]
# out_sample_tensor = transformer.sample(temp_keystroke)
# print(f"out_sample_tensor shape: {out_sample_tensor.shape}") # [bs, 128, 256]

# recon_emgs = vq_decoder(vq_latents)
# print(f"decoder_out shape: {recon_emgs.shape}") # [bs, seq_len, 6]
