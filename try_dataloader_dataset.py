import os
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import pickle as pkl
from tqdm import tqdm
import argparse

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
    # opt.epoch = 2
    opt.batch_size = 8
    opt.num_threads = 0
    # opt.config_file = 'config.yaml'
    # opt.learning_rate = 1e-05
    # opt.dynamic_lr = False
    # opt.dataroot = './dataset'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_num = torch.cuda.device_count()
    # configs = load_config(default(opt.config_file, opt.model_config_file))
    # if opt.dynamic_lr:
    #     base_lr = configs.model.base_learning_rate
    #     opt.learning_rate = base_lr * opt.batch_size * opt.acumulate_batch_size * len(opt.gpus)
    parser.print_options(opt)
    return opt, device, device_num


def train_and_val(model, train_loader, val_loader, optimizer, loss_fn, scheduler, opt, device):
    model.train()
    total_loss = 0
    for data in tqdm(train_loader):
        data['keystroke'] = data['keystroke'].to(device, dtype=torch.float32)
        # print(f"keystroke in train_loader shape: {data['keystroke'].shape}")
        data['emg'] = data['emg'].to(device, dtype=torch.float32)
        # print(f"emg in train_loader shape: {data['emg'].shape}")

        optimizer.zero_grad()
        out = model(data['keystroke'], data['emg'])
        # print(f"out shape: {out.shape}")
        loss = loss_fn(data['emg'], out)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data in tqdm(val_loader):
            data['keystroke'] = data['keystroke'].to(device, dtype=torch.float32)
            data['emg'] = data['emg'].to(device, dtype=torch.float32)
            out = model.sample(data['keystroke'])
            print(f"out in val shape: {out.shape}")
            loss = loss_fn(data['emg'], out)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    scheduler.step()

    return avg_train_loss, avg_val_loss

def main():
    opt, device, device_num = get_configurations()
    train_dataset = KeyEmgDataset(mode="train", win_len=1024, overlap_len=64)
    val_dataset = KeyEmgDataset(mode="val", win_len=8, overlap_len=0)
    train_loader = create_dataloader(train_dataset, opt, "KeyEmgDataloader", device_num, True, opt.batch_size)
    val_loader = create_dataloader(val_dataset, opt, "KeyEmgDataloader", device_num, False, opt.batch_size_val)
    # print(f"dataset len / batch_size = dataloader len: {len(train_dataset)} / {opt.batch_size} = {len(train_loader)}")

    print(f"running on device: {device}")
    args = get_network_parser()
    model = TransformerV2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    for epoch in range(opt.epoch):
        train_loss, val_loss = train_and_val(model, train_loader, val_loader, optimizer, loss_fn, scheduler, opt, device)
        print(f"Epoch {epoch + 1}/{opt.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


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


