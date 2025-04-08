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

from networks.transformer_lrf_VQ import TransformerV2
from networks.vq_lrf import VQEncoderV3, VQDecoderV3, Quantizer, VQTokenizerTrainerV3, TransformerT2MTrainer
from dataloader.dataset import KeyEmgDataset, create_dataloader
from cfg import config
from cfg.options import Options, TrainVQTokenizerOptions, TrainVQTransformerOptions

def get_network_parser():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--config', type=str, default='./cfg/emg_former_cfg.yaml', help='config file')
    parser.add_argument('opts', help=' ', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def get_configurations():
    parser = TrainVQTransformerOptions()
    opt = parser.get_options()
    
    opt.is_continue = False

    opt.dim_vq_latent = 256
    opt.en_channels = [128, 256, opt.dim_vq_latent]
    opt.de_channels = [opt.dim_vq_latent, 256, 128, 6]
    opt.train_files_num = 6000
    opt.val_files_num = 500
    opt.mode = "train"
    opt.precision = "bf16"
    opt.epoch = 100
    opt.batch_size = 64
    opt.batch_size_val = 64
    opt.log_rate = 200
    opt.win_len = 1024
    opt.win_len_val = 1024
    opt.overlap = 0
    opt.overlap_val = 0

    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_num = torch.cuda.device_count()
    parser.print_options(opt)
    return opt, opt.device, device_num

def plot_recon_emg(emg_data, pred_data, keystroke, epoch, opt, mode):
    batch_size, seq_len, emg_channels, keystroke_channels = emg_data.shape[0], emg_data.shape[1], emg_data.shape[2], keystroke.shape[2]

    for batch_idx in range(min(batch_size, 15)):
        fig, axs = plt.subplots(3, 1, figsize=(15, 15))
        # Plot Keystroke Data
        for channel in range(keystroke_channels):
            if torch.mean(keystroke[batch_idx, :, channel]) != -1:
                axs[0].plot(range(seq_len), keystroke[batch_idx, :, channel].cpu().detach().numpy(), label=f'Key {channel+1}')
        axs[0].set_title('Keystroke')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Height')
        axs[0].legend()
        axs[0].set_ylim(-1, 1)

        # Plot Ground Truth EMG Data
        for channel in range(emg_channels):
            axs[1].plot(range(seq_len), emg_data[batch_idx, :, channel].cpu().detach().numpy(), label=f'Muscle {channel+1}')
        axs[1].set_title('GT EMG')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Amplitude')
        axs[1].legend()
        axs[1].set_ylim(-1, 1)

        # Plot Predicted EMG Data
        for channel in range(emg_channels):
            axs[2].plot(range(seq_len), pred_data[batch_idx, :, channel].cpu().detach().numpy(), label=f'Muscle {channel+1}')
        axs[2].set_title('Pred EMG')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Amplitude')
        axs[2].legend()
        axs[2].set_ylim(-1, 1)

        plt.tight_layout()
        # wandb.log({f"Epoch {epoch} Sample {batch_idx}": wandb.Image(fig)})
        if mode == "train":
            save_dir = pjoin(opt.train_path, 'E%04d' % (epoch))
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = pjoin(opt.eval_path, 'E%04d' % (epoch))
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"Epoch_{epoch}-Sample_{batch_idx}.png"))
        plt.close(fig)


def load_models(opt):
    vq_encoder = VQEncoderV3(input_size=6, channels=opt.en_channels, n_down=3)
    vq_decoder = VQDecoderV3(input_size=opt.dim_vq_latent, channels=opt.de_channels, n_resblk=2, n_up=3)
    quantizer = Quantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)

    vq_checkpoint = torch.load('../../Piano_EMG_NIPS25_checkpoints/VQ_Model/emg_VQ/2025-04-04_18-02-21 128x256/finest.tar',
                            map_location='cuda')
    vq_encoder.load_state_dict(vq_checkpoint['vq_encoder'])    
    vq_decoder.load_state_dict(vq_checkpoint['vq_decoder'])
    quantizer.load_state_dict(vq_checkpoint['quantizer'])
    return vq_encoder, vq_decoder, quantizer


def train_and_val(model, vq_encoder, vq_decoder, quantizer, train_loader, val_loader, optimizer, loss_fn, scheduler, opt, epoch, device):
    model.train()
    total_loss = 0
    # with accelerator.autocast():
    for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{opt.epoch}", unit="batch")):
        # print(f"keystroke in train_loader shape: {data['keystroke'].shape}")
        # print(f"emg in train_loader shape: {data['emg'].shape}")
        optimizer.zero_grad()
        data['keystroke'], data['emg'] = data['keystroke'].to(device), data['emg'].to(device)

        pre_latents = vq_encoder(data['emg'])
        # print(f"encoder_out shape: {pre_latents.shape}") # [bs, 128, opt.dim_vq_latent]
        embedding_loss, vq_latents, _, perplexity = quantizer(pre_latents)
        # print(f"embedding_loss: {embedding_loss}") # []
        # print(f"vq_latents shape: {vq_latents.shape}") # [bs, 128, opt.dim_vq_latent]   
        # print(f"perplexity: {perplexity}") # 71.68  

        out = model(data['keystroke'], vq_latents) # [bs, 128, opt.dim_vq_latent]
        emb_loss = loss_fn(vq_latents, out)
        wandb.log({'train_emb_loss': emb_loss.item()})
        
        recon_emgs = vq_decoder(out)
        # print(f"decoder_out shape: {recon_emgs.shape}") # [bs, seq_len, 6]
        recon_loss = loss_fn(data['emg'], recon_emgs)
        wandb.log({'train_recon_loss': recon_loss.item()})
        loss = emb_loss + recon_loss

        # out = model.sample(data['keystroke'])
        # print(f"out shape: {out.shape}")

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        # if accelerator.is_main_process:
        wandb.log({'train_loss': loss.item()})
        if i % opt.log_rate == 0 and epoch % 2 == 0:
            plot_recon_emg(data['emg'], recon_emgs, data['keystroke'], epoch, opt, "train")
    
    avg_train_loss = total_loss / len(train_loader)
    # if accelerator.is_main_process:
    wandb.log({'avg_train_loss': avg_train_loss})

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        # with accelerator.autocast():
        for i, data in enumerate(tqdm(val_loader, desc="Validating", unit="batch")):
            data['keystroke'], data['emg'] = data['keystroke'].to(device), data['emg'].to(device)
            pre_latents = vq_encoder(data['emg'])
            embedding_loss, vq_latents, _, perplexity = quantizer(pre_latents)

            out = model.sample(data['keystroke']) # [bs, 128, opt.dim_vq_latent]
            emb_loss = loss_fn(vq_latents, out)
            wandb.log({'val_emb_loss': emb_loss.item()})
            recon_emgs = vq_decoder(out)
            recon_loss = loss_fn(data['emg'], recon_emgs)
            wandb.log({'val_recon_loss': recon_loss.item()})
            loss = emb_loss + recon_loss

            total_val_loss += loss.item()
            # if accelerator.is_main_process:
            wandb.log({'val_loss': loss.item()})
            if i % opt.log_rate == 0 and epoch % 2 ==0:
                plot_recon_emg(data['emg'], recon_emgs, data['keystroke'], epoch, opt, "val")

    avg_val_loss = total_val_loss / len(val_loader)
    # if accelerator.is_main_process:
    wandb.log({'avg_val_loss': avg_val_loss})
    scheduler.step()

    return avg_train_loss, avg_val_loss




if __name__ == '__main__':
    opt, device, device_num = get_configurations()
    args = get_network_parser()

    logging.basicConfig(
        filename=os.path.join(opt.ckpt_path, "train.log"),  
        filemode="a",           
        level=logging.INFO,     
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    train_dataset = KeyEmgDataset(mode="train", win_len=opt.win_len, overlap_len=opt.overlap)
    val_dataset = KeyEmgDataset(mode="val", win_len=opt.win_len_val, overlap_len=opt.overlap_val)
    train_loader = create_dataloader(train_dataset, opt, "KeyEmgDataloader", device_num, True, opt.batch_size)
    print(f"train dataset len / batch_size = dataloader len: {len(train_dataset)} / {opt.batch_size} = {len(train_loader)}")
    val_loader = create_dataloader(val_dataset, opt, "KeyEmgDataloader", device_num, False, opt.batch_size_val)
    print(f"val dataset len / batch_size = dataloader len: {len(val_dataset)} / {opt.batch_size_val} = {len(val_loader)}")


    vq_encoder, vq_decoder, quantizer = load_models(opt)
    vq_encoder.to(device)
    vq_decoder.to(device)
    quantizer.to(device)
    vq_encoder.eval()
    vq_decoder.eval()
    quantizer.eval()

    model = TransformerV2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    wandb.init(project='Piano_EMG_NIPS25_VQ_transformer', name='all_data_on_4090_128x256')
    wandb.watch(
            models=[model],
            criterion=None,
            log="all",
            log_freq=1
        )


    best_val_loss = float('inf')
    for epoch in range(1, opt.epoch+1):
        train_loss, val_loss = train_and_val(model, vq_encoder, vq_decoder, quantizer, train_loader, val_loader, optimizer, loss_fn, scheduler, opt, epoch, device)
        print(f"Epoch {epoch}/{opt.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logging.info(f"Epoch {epoch}/{opt.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # if accelerator.is_main_process:
            print(f'Best model saved at epoch {epoch}, with val_loss {best_val_loss}')
            logging.info(f'Best model saved at epoch {epoch}, with val_loss {best_val_loss}')
            torch.save(model.state_dict(), os.path.join(opt.ckpt_path,'best_epoch_'+str(epoch)+'.pth'))
        else:
            print(f"Not saved at epoch {epoch}, current val_loss is {val_loss}")
        # if accelerator.is_main_process:
        torch.save(model.state_dict(), os.path.join(opt.ckpt_path, 'latest_epoch.pth'))

    torch.save(model.state_dict(), os.path.join(opt.ckpt_path,'final_epoch_'+str(epoch)+'.pth'))
    wandb.finish()

