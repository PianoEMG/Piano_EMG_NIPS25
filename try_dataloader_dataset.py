import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
# import numpy as np
import pickle as pkl
from tqdm import tqdm
import argparse
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import wandb
import matplotlib.pyplot as plt
import logging

from networks.transformer_lrf import TransformerV2
from dataloader.dataset import KeyEmgDataset, create_dataloader
from cfg import config
from cfg.options import Options

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
    parser = Options(eval=False)
    opt = parser.get_options()
    opt.mode = "train"
    opt.precision = "bf16"
    opt.epoch = 100
    opt.batch_size = 8
    opt.batch_size_val = 8
    opt.log_rate = 100
    opt.win_len = 1024
    opt.win_len_val = 1024
    opt.overlap = 0
    opt.overlap_val = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_num = torch.cuda.device_count()
    parser.print_options(opt)
    return opt, device, device_num

def log_and_plot_batch(emg_data, pred_data, keystroke, epoch, opt, mode):
    batch_size, seq_len, emg_channels, keystroke_channels = emg_data.shape[0], emg_data.shape[1], emg_data.shape[2], keystroke.shape[2]
    # print(batch_size, seq_len, emg_channels, keystroke_channels)

    for batch_idx in range(batch_size):
        fig, axs = plt.subplots(3, 1, figsize=(15, 15))
        # Plot Keystroke Data
        for channel in range(keystroke_channels):
            if torch.mean(keystroke[batch_idx, :, channel]) != -1:
                axs[0].plot(range(seq_len), keystroke[batch_idx, :, channel].cpu().detach().numpy(), label=f'Key {channel+1}')
        axs[0].set_title('Keystroke')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Pressed_Depth')
        axs[0].legend()

        # Plot Ground Truth EMG Data
        for channel in range(emg_channels): 
            axs[1].plot(range(seq_len), emg_data[batch_idx, :, channel].cpu().detach().numpy(), label=f'Muscle {channel+1}')
        axs[1].set_title('GT EMG')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Amplitude')
        axs[1].legend()

        # Plot Predicted EMG Data
        for channel in range(emg_channels):
            axs[2].plot(range(seq_len), pred_data[batch_idx, :, channel].cpu().detach().numpy(), label=f'Muscle {channel+1}')
        axs[2].set_title('Pred EMG')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Amplitude')
        axs[2].legend()

        plt.tight_layout()
        # wandb.log({f"Epoch {epoch} Sample {batch_idx}": wandb.Image(fig)})
        plt.savefig(os.path.join(opt.log_path, f"{mode}_Epoch_{epoch+1}-Batch_{batch_idx+1}"))
        plt.close(fig)


def train_and_val(accelerator, model, train_loader, val_loader, optimizer, loss_fn, scheduler, opt, epoch):
    model.train()
    total_loss = 0
    with accelerator.autocast():
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epoch}", unit="batch")):
            # print(f"keystroke in train_loader shape: {data['keystroke'].shape}")
            # print(f"emg in train_loader shape: {data['emg'].shape}")
            optimizer.zero_grad()
            out = model(data['keystroke'], data['emg'])
            # print(f"out shape: {out.shape}")
            loss = loss_fn(data['emg'], out)
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
            if accelerator.is_main_process:
                wandb.log({'train_loss': loss.item()})
                if i % opt.log_rate == 0:
                    log_and_plot_batch(data['emg'], out, data['keystroke'], epoch, opt, "train")
    
    avg_train_loss = total_loss / len(train_loader)
    if accelerator.is_main_process:
        wandb.log({'avg_train_loss': avg_train_loss})

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        with accelerator.autocast():
            for i, data in enumerate(tqdm(val_loader, desc="Validating", unit="batch")):
                out = model.sample(data['keystroke'])
                # print(f"out in val shape: {out.shape}")
                loss = loss_fn(data['emg'], out)
                total_val_loss += loss.item()
                if accelerator.is_main_process:
                    wandb.log({'val_loss': loss.item()})

                if i % opt.log_rate == 0 and accelerator.is_main_process:
                    # print(f"log_rate: {opt.log_rate}")
                    log_and_plot_batch(data['emg'], out, data['keystroke'], epoch, opt, "val")

    avg_val_loss = total_val_loss / len(val_loader)
    if accelerator.is_main_process:
        wandb.log({'avg_val_loss': avg_val_loss})
    scheduler.step()

    return avg_train_loss, avg_val_loss

def main():
    opt, device, device_num = get_configurations()
    args = get_network_parser()
    projection_config = ProjectConfiguration(project_dir=opt.ckpt_path)
    accelerator = Accelerator(mixed_precision=opt.precision, log_with="wandb", project_config=projection_config)

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

    print(f"running on device: {device}")
    model = TransformerV2()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader)
    if accelerator.is_main_process:
        wandb.init(project='Piano_EMG_NIPS25')
        accelerator.init_trackers(opt.name)
        wandb.watch(model)

    best_val_loss = 100.0
    if not os.path.exists(opt.ckpt_path):
        os.makedirs(opt.ckpt_path)
    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path)

    for epoch in range(opt.epoch):
        train_loss, val_loss = train_and_val(accelerator, model, train_loader, val_loader, optimizer, loss_fn, scheduler, opt, epoch)
        print(f"Epoch {epoch + 1}/{opt.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logging.info(f"Epoch {epoch + 1}/{opt.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if accelerator.is_main_process:
                print(f'Best model saved at epoch {epoch}, with val_loss {best_val_loss}')
                logging.info(f'Best model saved at epoch {epoch}, with val_loss {best_val_loss}')
                torch.save(model.state_dict(), os.path.join(opt.ckpt_path,'best_epoch_'+str(epoch)+'.pth'))
        else:
            print(f"Not saved at epoch {epoch}, current val_loss is {val_loss}")
        if accelerator.is_main_process:
            torch.save(model.state_dict(), os.path.join(opt.ckpt_path, 'latest_epoch.pth'))
    
    if accelerator.is_main_process:
        torch.save(model.state_dict(), os.path.join(opt.ckpt_path,'final_epoch_'+str(epoch)+'.pth'))
        wandb.finish()


main()

# with open(f'C:/Users/ruofa/Desktop/Piano_Dataset/keystroke_data/t4_p19_1.pkl', 'rb') as f2:
#     keystroke_data = pkl.load(f2)
#     keystroke_data = torch.tensor(keystroke_data).float()
# with open(f'C:/Users/ruofa/Desktop/Piano_Dataset/emg_data/t4_p19_1.pkl', 'rb') as f:
#     emg_data = pkl.load(f)
#     emg_data = torch.tensor(emg_data).float()
# keystroke_data, emg_data = np.arange(3905), np.arange(3905)
# print(keystroke_data.shape, emg_data.shape)
# num_windows = (keystroke_data.shape[0] - 64) // (1024 - 64) 
# split_keystroke = segment_sliding_subarray(keystroke_data, 1024, 64, num_windows, "train")
# print(split_keystroke.shape)

