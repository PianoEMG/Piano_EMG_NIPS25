import os
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import pickle as pkl
from tqdm import tqdm
import argparse
# from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import wandb
import matplotlib.pyplot as plt
import logging
from os.path import join as pjoin
from torch.utils.data import DataLoader
import wandb

from networks.transformer_lrf import TransformerV2
from networks.vq_lrf import VQEncoderV3, VQDecoderV3, Quantizer, VQTokenizerTrainerV3
from dataloader.dataset import KeyEmgDataset, create_dataloader
# from dataloader.motion_dataset import MotionDataset
from cfg import config
from cfg.options import Options, TrainVQTokenizerOptions

def get_configurations():
    parser = TrainVQTokenizerOptions()
    opt = parser.get_options()
    opt.train_files_num = 300
    opt.val_files_num = 10
    opt.mode = "train"
    opt.precision = "bf16"
    opt.epoch = 100
    opt.batch_size = 64
    opt.batch_size_val = 64
    opt.log_rate = 100
    opt.win_len = 1024
    opt.win_len_val = 1024
    opt.overlap = 0
    opt.overlap_val = 0

    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_num = torch.cuda.device_count()
    parser.print_options(opt)
    return opt, opt.device, device_num

def plot_recon_emg(data, save_dir):
    # data = train_dataset.inv_transform(data)
    # for i in range(len(data)):
    #     joint_data = data[i]
    #     joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
    #     save_path = pjoin(save_dir, '%02d.mp4' % (i))
    #     plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)
    os.makedirs(save_dir, exist_ok=True)
    batch_size = data.shape[0] // 2 
    recon_data, gt_emg = data[:batch_size], data[batch_size:]  
    for i in range(batch_size):
        save_path = pjoin(save_dir, '%02d.png' % (i))
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        axs[0].plot(range(data.shape[1]), gt_emg[i, :, :], label='GT EMG')
        axs[1].plot(range(data.shape[1]), recon_data[i, :, :], label='Recon EMG')
        axs[0].legend()
        axs[0].set_title('GT EMG')
        axs[1].legend()
        axs[1].set_title('Recon EMG')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
    return


def load_models(opt):
    # vq_encoder = VQEncoderV3(dim_pose - 4, enc_channels, opt.n_down)
    # vq_encoder = VQEncoderV3(input_size=6, channels= [1024, opt.dim_vq_latent], n_down=2)
    # vq_decoder = VQDecoderV3(input_size=opt.dim_vq_latent, channels=[opt.dim_vq_latent, 1024, 6], n_resblk=2, n_up=2)
    vq_encoder = VQEncoderV3(input_size=6, channels= [128, 256, 512, opt.dim_vq_latent], n_down=4)
    vq_decoder = VQDecoderV3(input_size=opt.dim_vq_latent, channels=[opt.dim_vq_latent, 512, 256, 128, 6], n_resblk=2, n_up=4)

    quantizer = Quantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)
    # discriminator = VQDiscriminator(6, opt.dim_vq_dis_hidden, opt.n_layers_dis)

    # if not opt.is_continue:
    #     checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name,
    #                                   'VQVAEV3_CB1024_CMT_H1024_NRES3', 'model', 'finest.tar'),
    #                             map_location=opt.device)
    #     vq_encoder.load_state_dict(checkpoint['vq_encoder'])
    #     vq_decoder.load_state_dict(checkpoint['vq_decoder'])
    #     quantizer.load_state_dict(checkpoint['quantizer'])
    return vq_encoder, vq_decoder, quantizer

if __name__ == '__main__':
    opt, device, device_num = get_configurations()
    # args = parser.parse()

    train_dataset = KeyEmgDataset(mode="train", win_len=opt.win_len, overlap_len=opt.overlap)
    val_dataset = KeyEmgDataset(mode="val", win_len=opt.win_len_val, overlap_len=opt.overlap_val)
    train_loader = create_dataloader(train_dataset, opt, "KeyEmgDataloader", device_num, True, opt.batch_size)
    print(f"train dataset len / batch_size = dataloader len: {len(train_dataset)} / {opt.batch_size} = {len(train_loader)}")
    val_loader = create_dataloader(val_dataset, opt, "KeyEmgDataloader", device_num, False, opt.batch_size_val)
    print(f"val dataset len / batch_size = dataloader len: {len(val_dataset)} / {opt.batch_size_val} = {len(val_loader)}")


    vq_encoder, vq_decoder, quantizer = load_models(opt)
    trainer = VQTokenizerTrainerV3(opt, vq_encoder, quantizer, vq_decoder)

    wandb.init(project='Piano_EMG_NIPS25_VQ', name='p1_data_on_Dell')
    wandb.watch(
            models=[vq_encoder, quantizer, vq_decoder],
            criterion=None,
            log="all",
            log_freq=1
        )

    trainer.train(train_loader, val_loader, plot_recon_emg)

    wandb.finish()


    # train_tensor = torch.randn(8, 1024, 6)
    # pre_latents = vq_encoder(train_tensor)
    # print(f"encoder_out shape: {pre_latents.shape}") # [bs, seq_len, 6]

    # embedding_loss, vq_latents, _, perplexity = quantizer(pre_latents)
    # print(f"embedding_loss: {embedding_loss}") # []
    # print(f"vq_latents shape: {vq_latents.shape}") # [bs, 256, 6]   
    # print(f"perplexity: {perplexity}") # 71.68    

    # recon_motions = vq_decoder(vq_latents)
    # print(f"decoder_out shape: {recon_motions.shape}") # [bs, seq_len, 6]

