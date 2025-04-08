import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base_models import Transformer, LinearEmbedding, PositionalEncoding

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_channels, out_channels)
        
        # Adjust for residual connection
        self.residual_connection = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual_connection(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
            # if i < num_levels - 1:
            #     layers += [nn.MaxPool1d(kernel_size=2, stride=2)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y1 = self.tcn(x)
        o = y1.permute(0, 2, 1)
        return o

# CHI Transformer Model for Seq2Seq
class EMGFormer(nn.Module):
    def __init__(self, in_size=88,
               out_size=6,
               hidden_size=1024,
               TCN_hidden_size=512,
               num_hidden_layers=3,
               num_attention_heads=4,
               intermediate_size=512,
               is_compressed=False):
        super(EMGFormer, self).__init__()

        self.TCN = TCN(input_size=in_size, output_size=in_size, num_channels=[256, 256, 256, 256, TCN_hidden_size], kernel_size=3,dropout=0.2)
        
        self.transformer = Transformer(in_size=hidden_size,
                                       hidden_size=hidden_size,
                                       num_hidden_layers=num_hidden_layers,
                                       num_attention_heads=num_attention_heads,
                                       intermediate_size=intermediate_size,
                                       is_compressed=is_compressed)
        self.encoder_pos_embedding = PositionalEncoding(hidden_size)
        self.encoder_linear_embedding = LinearEmbedding(hidden_size, hidden_size)
        self.out_fc = nn.Linear(hidden_size, out_size)
        # self.upsample = nn.Linear(128, 2048)
        # self.transformer = SoleFormer_Model(input_dim=hidden_size, d_model=768, nhead=4, num_encoder_layers=3, output_dim=64, is_seq2seq=True)

        # self.out = LinearEmbedding(hidden_size, out_size)

    def forward(self, x):
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        x = x.permute(0, 2, 1)
        x = self.TCN(x) 
        print(f"x after TCN shape: {x.shape}")
        x = x.permute(0, 2, 1)

        # x = self.encoder_linear_embedding(x)
        x = self.encoder_pos_embedding(x)
        print(f"x after pos embedding shape: {x.shape}")
        x = self.transformer((x, dummy_mask))
        # print(f"x after transformer shape: {x.shape}")
        x = self.out_fc(x)
        # x = x.permute(2, 1, 0)
        # x = self.upsample(x)
        
        # print('x trans', x.shape)
        x = x.permute(1, 0, 2)
        # x = x.permute(0, 2, 1)
        # print('x trans', x.shape)
        

        return x


# Basic Transformer Model for Seq2Seq Using nn.TransformerEncoder
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 input_size=88, output_size=6,
                 d_model=512,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        
        self.output_proj= nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        output = self.output_proj(x)  # [batch_size, seq_len, input_size]
        
        return output