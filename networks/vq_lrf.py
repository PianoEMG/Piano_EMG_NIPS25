# copied from T2MT: https://github.com/EricGuo5513/TM2T/blob/main/networks/modules.py
import torch
import torch.nn as nn
import numpy as np
import time
import math
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
import torch.optim as optim
from os.path import join as pjoin
import logging
import wandb

from networks.layers import *
from utils.utils import *

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class VQEncoderV3(nn.Module):
    def __init__(self, input_size, channels, n_down):
        super(VQEncoderV3, self).__init__()
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return outputs



class VQDecoderV3(nn.Module):
    def __init__(self, input_size, channels, n_resblk, n_up):
        super(VQDecoderV3, self).__init__()
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs



class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q - z.detach())**2) + self.beta * \
               torch.mean((z_q.detach() - z)**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))
        return loss, z_q, min_encoding_indices, perplexity

    def map2index(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def get_codebook_entry(self, indices):
        """

        :param indices(B, seq_len):
        :return z_q(B, seq_len, e_dim):
        """
        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim, )).contiguous()
        return z_q
    



# import tensorflow as tf
# class Logger(object):
#     def __init__(self, log_dir):
#         self.writer = tf.summary.create_file_writer(log_dir)

#     def scalar_summary(self, tag, value, step):
#         with self.writer.as_default():
#             tf.summary.scalar(tag, value, step=step)
#             self.writer.flush()



class Trainer(object):
    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def ones_like(self, tensor, val=1.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)

    # @staticmethod
    def zeros_like(self, tensor, val=0.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)

    def forward(self, batch_data):
        pass

    def backward(self):
        pass

    def update(self):
        pass



class VQTokenizerTrainerV3(Trainer):
    def __init__(self, args, vq_encoder, quantizer, vq_decoder, discriminator=None):
        self.opt = args
        self.vq_encoder = vq_encoder
        self.vq_decoder = vq_decoder
        self.quantizer = quantizer
        # self.mov_encoder = mov_encoder
        # self.discriminator = discriminator
        self.device = args.device

        if args.is_train:
            # self.logger = Logger(args.log_dir)
            self.logger = logging.basicConfig(
                filename=os.path.join(args.ckpt_path, "train.log"),  
                filemode="a",           
                level=logging.INFO,     
                format="%(asctime)s - %(levelname)s - %(message)s"
            )
            # self.l1_criterion = torch.nn.L1Loss()
            self.loss_fn = torch.nn.MSELoss()
            # self.gan_criterion = torch.nn.BCEWithLogitsLoss()
            # self.disc_loss = self.hinge_d_loss

    # def hinge_d_loss(self, logits_real, logits_fake):
    #     loss_real = torch.mean(F.relu(1. - logits_real))
    #     loss_fake = torch.mean(F.relu(1. + logits_fake))
    #     d_loss = 0.5 * (loss_real + loss_fake)
    #     return d_loss

    # def ones_like(self, tensor, val=1.):
    #     return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)
    #
    # # @staticmethod
    # def zeros_like(self, tensor, val=0.):
    #     return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)

    def forward(self, batch_data):
        motions = batch_data["emg"]
        self.motions = motions.detach().to(self.device).float()
        # print(f"motions shape:{self.motions.shape}")
        self.pre_latents = self.vq_encoder(self.motions)
        # print(f"pre_latents shape:{self.pre_latents.shape}")
        self.embedding_loss, self.vq_latents, _, self.perplexity = self.quantizer(self.pre_latents)
        # print(f"vq_latents shape:{self.vq_latents.shape}")
        self.recon_motions = self.vq_decoder(self.vq_latents)

    # def calculate_adaptive_weight(self, rec_loss, gan_loss, last_layer):
    #     rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
    #     gan_grads = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]

    #     d_weight = torch.norm(rec_grads) / (torch.norm(gan_grads) + 1e-4)
    #     d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    #     d_weight = d_weight * self.opt.lambda_adv
    #     return d_weight


    def backward_G(self):
        self.loss_rec_mot = self.loss_fn(self.recon_motions, self.motions)
        self.loss_G = self.loss_rec_mot + self.embedding_loss

        # if self.opt.start_use_gan:
        #     _, logits_fake = self.discriminator(self.recon_motions)
        #     self.loss_G_adv = -torch.mean(logits_fake)
        #     # last_layer = self.vq_decoder.main[9].weight
        #     #
        #     # try:
        #     #     self.d_weight = self.calculate_adaptive_weight(self.loss_rec_mot, self.loss_G_adv, last_layer=last_layer)
        #     # except RuntimeError:
        #     #     assert not self.opt.is_train
        #     #     self.d_weight = torch.tensor(0.0)
        #     # self.loss_G += self.d_weight * self.loss_G_adv
        #     self.loss_G += self.opt.lambda_adv * self.loss_G_adv


    # def backward_D(self):
    #     self.real_feats, real_labels = self.discriminator(self.motions.detach())
    #     fake_feats, fake_labels = self.discriminator(self.recon_motions.detach())

    #     self.loss_D = self.disc_loss(real_labels, fake_labels) * self.opt.lambda_adv
    #     # self.loss_D = (self.loss_D_T + self.loss_D_F) * self.opt.lambda_adv


    def update(self):
        loss_logs = OrderedDict({})

        # if self.opt.start_use_gan:
        #     self.zero_grad([self.opt_discriminator])
        #     self.backward_D()
        #     self.loss_D.backward(retain_graph=True)
        #     self.step([self.opt_discriminator])

        self.zero_grad([self.opt_vq_encoder, self.opt_quantizer, self.opt_vq_decoder])
        self.backward_G()
        self.loss_G.backward()
        self.step([self.opt_vq_encoder, self.opt_quantizer, self.opt_vq_decoder])

        loss_logs['loss_G'] = self.loss_G.item()
        loss_logs['loss_G_rec_emg'] = self.loss_rec_mot.item()
        loss_logs['loss_G_emb'] = self.embedding_loss.item()
        loss_logs['perplexity'] = self.perplexity.item()
        
        wandb.log({'loss_G': self.loss_G.item()})
        wandb.log({'loss_G_rec_emg': self.loss_rec_mot.item()})
        wandb.log({'loss_G_emb': self.embedding_loss.item()})

        # if self.opt.start_use_gan:
        #     # loss_logs['d_weight'] = self.d_weight.item()
        #     loss_logs['loss_G_adv'] = self.loss_G_adv.item()
        #     loss_logs['loss_D'] = self.loss_D.item()

        return loss_logs

    def save(self, file_name, ep, total_it):
        state = {
            'vq_encoder': self.vq_encoder.state_dict(),
            'quantizer': self.quantizer.state_dict(),
            'vq_decoder': self.vq_decoder.state_dict(),
            # 'discriminator': self.discriminator.state_dict(),

            'opt_vq_encoder': self.opt_vq_encoder.state_dict(),
            'opt_quantizer': self.opt_quantizer.state_dict(),
            'opt_vq_decoder': self.opt_vq_decoder.state_dict(),
            # 'opt_discriminator': self.opt_discriminator.state_dict(),

            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_encoder.load_state_dict(checkpoint['vq_encoder'])
        self.quantizer.load_state_dict(checkpoint['quantizer'])
        self.vq_decoder.load_state_dict(checkpoint['vq_decoder'])

        self.opt_vq_encoder.load_state_dict(checkpoint['opt_vq_encoder'])
        self.opt_quantizer.load_state_dict(checkpoint['opt_quantizer'])
        self.opt_vq_decoder.load_state_dict(checkpoint['opt_vq_decoder'])

        # if self.opt.use_gan:
        # self.discriminator.load_state_dict(checkpoint['discriminator'])
        # self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        return checkpoint['ep'], checkpoint['total_it']

    # def train(self, train_dataloader, val_dataloader, plot_eval):
    def train(self, train_dataloader, val_dataloader, plot_eval):
        self.vq_encoder.to(self.device)
        self.quantizer.to(self.device)
        self.vq_decoder.to(self.device)
        # self.discriminator.to(self.device)

        self.opt_vq_encoder = optim.Adam(self.vq_encoder.parameters(), lr=self.opt.lr)
        self.opt_quantizer = optim.Adam(self.quantizer.parameters(), lr=self.opt.lr)
        self.opt_vq_decoder = optim.Adam(self.vq_decoder.parameters(), lr=self.opt.lr)
        # self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.ckpt_path, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict()
        while epoch < self.opt.epoch:
            self.opt.start_use_gan = (epoch >= self.opt.start_dis_epoch)
            for i, batch_data in enumerate(train_dataloader):
                self.vq_encoder.train()
                self.quantizer.train()
                self.vq_decoder.train()
                # if self.opt.use_percep:
                #     self.mov_encoder.train()
                # if self.opt.start_use_gan:
                #     # print('Introducing Adversarial Loss!~')
                #     self.discriminator.train()

                self.forward(batch_data)

                log_dict = self.update()
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    logging.info(f'Validation Loss: {val_loss}, it: {it}')

                    for tag, value in logs.items():
                        logging.info(f'{tag}: {value / self.opt.log_every}, it: {it}')
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.ckpt_path, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.ckpt_path, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.ckpt_path, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss_rec = 0
            val_loss_emb = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    val_loss_rec += self.loss_fn(self.recon_motions, self.motions).item()
                    val_loss_emb += self.embedding_loss.item()   

            val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)
            val_loss = val_loss_rec
            wandb.log({'val_loss_rec': val_loss_rec})
            wandb.log({'val_loss_emb': val_loss_emb})
            # wandb.log({'Validation Loss': val_loss})

            print('Validation Loss: %.5f' % (val_loss))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.ckpt_path, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            if epoch % self.opt.eval_every_e == 0:
                data = torch.cat([self.recon_motions, self.motions], dim=0).detach().cpu().numpy()
                print(f"data shape:{data.shape}")
                save_dir = pjoin(self.opt.eval_path, 'E%04d' % (epoch))
                os.makedirs(save_dir, exist_ok=True)
                plot_eval(data, save_dir)

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break
