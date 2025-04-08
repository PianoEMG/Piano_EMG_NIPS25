# copied from T2MT: https://github.com/EricGuo5513/TM2T/blob/main/networks/transformer.py
import torch
import torch.nn as nn
import numpy as np
from networks.layers import *

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def get_pad_mask(batch_size, seq_len, non_pad_lens):
    non_pad_lens = non_pad_lens.data.tolist()
    mask_2d = torch.zeros((batch_size, seq_len), dtype=torch.float16)
    for i, cap_len in enumerate(non_pad_lens):
        mask_2d[i, :cap_len] = 1
    return mask_2d.unsqueeze(1).bool()

def get_pad_mask_idx(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1)

def get_subsequent_mask(seq):
    # sz_b, seq_len = seq.shape
    sz_b, seq_len = seq.shape[0], seq.shape[1]
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len)), diagonal=1)).bool()
    return subsequent_mask.to(seq.device)


# def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
def cal_performance(pred, gold, smoothing=False):
    # loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)
    loss = cal_loss(pred, gold, smoothing=smoothing)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    # non_pad_mask = gold.ne(trg_pad_idx)
    non_pad_mask = torch.full_like(gold, True, dtype=torch.bool)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    # pred = pred.masked_select(non_pad_mask)
    return loss, pred, n_correct, n_word


# def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
def cal_loss(pred, gold, smoothing=False):
    '''Calculate cross entropy loss, apply label smoothing if needed.'''
    # gold = gold.contiguous().view(-1)

    # if smoothing:
    #     eps = 0.1
    #     n_class = pred.size(1)

    #     one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    #     one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class-1)
    #     log_prb = F.log_softmax(pred, dim=1)

    #     non_pad_mask = gold.ne(trg_pad_idx)
    #     loss = -(one_hot * log_prb).sum(dim=1)
    #     loss = loss.masked_select(non_pad_mask).sum()
    # else:
    loss = F.cross_entropy(pred, gold, reduction='sum')
    return loss


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.register_buffer('positional_encoding', self.encoding)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        return self.encoding[:seq_len, :].clone().detach().to(x.device) + x


"""Of which the inputs are tokens"""
class EncoderV2(nn.Module):
    def __init__(self, input_dim, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner,
                 dropout=0.1, n_position=40):
        super(EncoderV2, self).__init__()
        self.position_enc = PositionalEncoding(d_model, max_len=n_position)
        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.input_proj = nn.Linear(input_dim, d_word_vec, bias=False)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False, input_onehot=False):
        enc_slf_attn_list = []
        src_seq = self.input_proj(src_seq)
        src_seq *= self.d_model ** 0.5
        enc_output = self.position_enc(src_seq)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


"""Of which the inputs are tokens, outputs are discrete probablities"""
class Decoder(nn.Module):
    def __init__(self, output_dim, d_word_vec, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, n_position=200, dropout=0.1):
        super(Decoder, self).__init__()
        # self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.output_proj = nn.Linear(output_dim, d_word_vec, bias=False)
        self.position_enc = PositionalEncoding(d_word_vec, max_len=n_position)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = self.output_proj(trg_seq)
        dec_output *= self.d_model ** 0.5

        dec_output = self.position_enc(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,



"""Of which the source sequence is tokens, and the target input is token, output is discrete probs"""
"""Pretrained Word Embeddings are not used"""
class TransformerV2(nn.Module):
    def __init__(self, n_src_vocab=88, n_trg_vocab=256, 
                 d_src_word_vec=256, d_trg_word_vec=256,
                 d_model=256, d_inner=2048, n_enc_layers=6, n_dec_layers=6, n_head=8, d_k=64, d_v=64,
                 dropout=0.1, n_src_position=1024, n_trg_position=1024, trg_emb_prj_weight_sharing=False):
        super(TransformerV2, self).__init__()

        self.d_model = d_model
        self.temporal_down = nn.Sequential(
            nn.Conv1d(n_src_vocab, n_src_vocab, kernel_size=4, stride=2, padding=1),  # 1024 -> 512
            nn.BatchNorm1d(n_src_vocab),
            nn.ReLU(),
            nn.Conv1d(n_src_vocab, n_src_vocab, kernel_size=4, stride=2, padding=1),  # 512 -> 256
            nn.BatchNorm1d(n_src_vocab),
            nn.ReLU(),
            nn.Conv1d(n_src_vocab, n_src_vocab, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm1d(n_src_vocab),
            nn.ReLU()
        )
        self.linear_key2emg = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            zero_module(nn.Linear(d_model, out_features=d_model))
        )
        self.encoder = EncoderV2(
            input_dim=n_src_vocab, n_position=n_src_position, d_word_vec=d_src_word_vec,
            d_model=d_model, d_inner=d_inner, n_layers=n_enc_layers, n_head=n_head, d_k=d_k,
            d_v=d_v, dropout=dropout
        )
        self.decoder = Decoder(
            output_dim=n_trg_vocab, n_position=n_trg_position, d_word_vec=d_trg_word_vec,
            d_model=d_model, d_inner=d_inner, n_layers=n_dec_layers, n_head=n_head, d_k=d_k,
            d_v=d_v, dropout=dropout
        )
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

        if trg_emb_prj_weight_sharing:
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq, input_onehot=False, src_mask=None, src_non_pad_lens=None):
        # print("---------------------------now in forward-------------------------------")
        # print(f"src_seq, trg_seq shape: {src_seq.shape}, {trg_seq.shape}") #src_seq, trg_seq.shape: [batch_size, src_seq_len, 88], [batch_size, src_seq_len, 6]  
        src_seq = src_seq.permute(0, 2, 1)  # (bs, 88, 1024)
        src_seq = self.temporal_down(src_seq)  # (bs, 88, 128)
        src_seq = src_seq.permute(0, 2, 1)  # (bs, 128, 88)
        # src_seq = self.linear_key2emg(src_seq)
        # print(f"src_seq shape: {src_seq.shape}") # [bs, 128, 88]
        # trg_seq= torch.cat((src_seq[:, 0, :].unsqueeze(1), trg_seq[:, :-1, :]), dim=1)

        batch_size, src_seq_len = src_seq.shape[0], src_seq.shape[1]
        # print(f"trg_seq shape: {trg_seq.shape}") # [bs, seq_len, 6]
        src_non_pad_lens = torch.full((batch_size, ), src_seq_len)
        src_mask = get_pad_mask(batch_size, src_seq_len, src_non_pad_lens).to(src_seq.device)
        # print(f"src_mask shape: {src_mask.shape}") # [bs, 1, seq_len]
        
        trg_mask = get_subsequent_mask(trg_seq)
        # print(f"trg_mask shape: {trg_mask.shape}") # [1, seq_len, seq_len]

        enc_output, *_ = self.encoder(src_seq, src_mask, input_onehot, input_onehot=input_onehot)
        # print(f"enc_output shape: {enc_output.shape}") # [batch_size, 128, 512]   
        enc_output = self.linear_key2emg(enc_output)
        print(f"enc_output after linear_key2emg shape: {enc_output.shape}") # [batch_size, 128, 128]
        trg_seq= torch.cat((enc_output[:, 0, :].unsqueeze(1), trg_seq[:, :-1, :]), dim=1)
        # print(f"trg_seq shape: {trg_seq.shape}") # [batch_size, 1, 128]
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        # print(f"dec_output shape: {dec_output.shape}") # [batch_size, 128, 512]
        seq_logit = self.trg_word_prj(dec_output)
        # print(f"seq_logit shape: {seq_logit.shape}") # [batch_size, 128, 256]
        return seq_logit

    def sample_original(self, src_seq, trg_sos, trg_eos, max_steps=80, sample=False, top_k=None):
        trg_seq = torch.LongTensor(src_seq.size(0), 1).fill_(trg_sos).to(src_seq).long()
        # batch_size, src_seq_len = src_seq.shape[0], src_seq.shape[1]
        # src_mask = get_pad_mask(batch_size, src_seq_len, src_non_pad_lens).to(src_seq.device)
        src_mask = get_pad_mask_idx(src_seq, self.src_pad_idx)
        enc_output, *_ = self.encoder(src_seq, src_mask)

        for _ in range(max_steps):
            trg_mask = get_subsequent_mask(trg_seq)

            dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
            seq_logit = self.trg_word_prj(dec_output)
            logits = seq_logit[:, -1, :]

            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            # print(probs.sort(dim=1)[:top_k])
            # print(torch.topk(probs, k=10, dim=-1))
            _, ix = torch.topk(probs, k=1, dim=-1)
            if ix[0] == trg_eos:
                break

            if sample:
                ix = torch.multinomial(probs, num_samples=1)
                while (ix[0] in [trg_sos, trg_eos]):
                    ix = torch.multinomial(probs, num_samples=1)
            trg_seq = torch.cat((trg_seq, ix), dim=1)
        return trg_seq
    
    def sample(self, src_seq, trg_sos=0, sample=False, top_k=None):
        # print("---------------------------now in sample-------------------------------")
        src_seq = src_seq.permute(0, 2, 1)  # (bs, 88, 1024)
        src_seq = self.temporal_down(src_seq)  # (bs, 88, 128)
        src_seq = src_seq.permute(0, 2, 1)  # (bs, 128, 88)
        # trg_seq = torch.LongTensor(src_seq.size(0), 1, 256).fill_(trg_sos).to(src_seq).float()
        
        batch_size, src_seq_len = src_seq.shape[0], src_seq.shape[1]
        src_non_pad_lens = torch.full((batch_size, ), src_seq_len)
        src_mask = get_pad_mask(batch_size, src_seq_len, src_non_pad_lens).to(src_seq.device)
        # print(f"src_mask: {src_mask}")
        # print(f"src_mask shape: {src_mask.shape}") # [bs, 1, seq_len]
        enc_output, *_ = self.encoder(src_seq, src_mask)
        enc_output = self.linear_key2emg(enc_output)
        print(f"enc_output after linear_key2emg shape: {enc_output.shape}") # [batch_size, 128, 128]
        trg_seq= enc_output[:, 0, :].unsqueeze(1)
        print(f"trg_seq shape: {trg_seq.shape}")  # [bs, 128]
        
        while trg_seq.shape[1] < src_seq.shape[1]:
            # print(f"trg_seq: {trg_seq}")
            # print(f"trg_seq shape: {trg_seq.shape}") # [bs, 1, 6] -> [bs, 2, 6] -> [bs, 3, 6]

            trg_mask = get_subsequent_mask(trg_seq)
            # print(f"trg_mask: {trg_mask}")
            # print(f"trg_mask shape: {trg_mask.shape}") # [1, 1, 1] -> [1, 2, 2] -> [1, 3, 3]

            dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
            # print(f"dec_output: {dec_output}")
            # print(f"dec_output shape: {dec_output.shape}") # [bs, 1, 512] -> [bs, 2, 512] -> [bs, 3, 512]

            seq_logit = self.trg_word_prj(dec_output)
            # print(f"seq_logit: {seq_logit}")
            # print(f"seq_logit shape: {seq_logit.shape}") # [bs, 1, 6] -> [bs, 2, 6] -> [bs, 3, 6]

            logits = seq_logit[:, -1, :]
            noise = torch.randn_like(logits) * 0.05
            logits = logits + noise
            logits = logits.unsqueeze(1)
            # print(f"logits shape: {logits.shape}")
            trg_seq = torch.cat((trg_seq, logits), dim=1)
        # print(f"trg_seq shape: {trg_seq.shape}")
        return trg_seq

