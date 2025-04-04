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
        # print('y1:', y1.shape)
        # o = self.linear(y1.permute(0, 2, 1))
        o = y1.permute(0, 2, 1)
        return o#.permute(0, 2, 1)

class EMGFormer(nn.Module):
    def __init__(self, in_size=88,
               out_size=6,
               hidden_size=512,
               num_hidden_layers=3,
               num_attention_heads=4,
               intermediate_size=512,
               is_compressed=False):
        super(EMGFormer, self).__init__()

        self.TCN = TCN(input_size=in_size, output_size=in_size, num_channels=[256, 256, 256, 256, hidden_size], kernel_size=3,dropout=0.2)
        
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
        # x [bs, seq, ch]
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        # x [bs, ch, seq]
        x = x.permute(0, 2, 1)
        x = self.TCN(x) 
        # print(f"x after TCN shape: {x.shape}")

        # x [bs, seq, ch]
        x = x.permute(1, 0, 2)
        # x [seq, bs, ch]
        # print('x permute', x.shape)
        x = self.encoder_linear_embedding(x)
        x = self.encoder_pos_embedding(x)
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

class EMGPoseFormer(nn.Module):
    def __init__(self, in_size=2334,
               out_size=64,
               hidden_size=512,
               num_hidden_layers=6,
               num_attention_heads=8,
               intermediate_size=1536,
               is_compressed=False):
        super(EMGPoseFormer, self).__init__()
        
        self.transformer = Transformer(in_size=hidden_size,
                                       hidden_size=hidden_size,
                                       num_hidden_layers=num_hidden_layers,
                                       num_attention_heads=num_attention_heads,
                                       intermediate_size=intermediate_size,
                                       is_compressed=is_compressed)
        self.encoder_pos_embedding = PositionalEncoding(hidden_size)
        self.encoder_linear_embedding = LinearEmbedding(in_size, hidden_size)
        self.out_fc = nn.Linear(hidden_size, out_size)
        # self.transformer = SoleFormer_Model(input_dim=hidden_size, d_model=768, nhead=4, num_encoder_layers=3, output_dim=64, is_seq2seq=True)

        # self.out = LinearEmbedding(hidden_size, out_size)

    def forward(self, x):
        # x [bs, seq, ch]
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}

        x = x.permute(1, 0, 2)
        # x [seq, bs, ch]
        # print('x permute', x.shape)
        x = self.encoder_linear_embedding(x)
        x = self.encoder_pos_embedding(x)
        x = self.transformer((x, dummy_mask))
        x = self.out_fc(x)
        
        # print('x trans', x.shape)
        x = x.permute(1, 0, 2)
        x = x.permute(0, 2, 1)
        # print('x trans', x.shape)
        
        return x
    
class EMGCrossFormer(nn.Module):
    def __init__(self, ks_size=88,
               pose_size=64,
               out_size=64,
               hidden_size=512,
               num_hidden_layers=3,
               num_attention_heads=4,
               intermediate_size=512,
               is_compressed=False):
        super(EMGCrossFormer, self).__init__()

        self.TCN = TCN(input_size=ks_size, output_size=ks_size, num_channels=[256, 256, 256, 256, hidden_size], kernel_size=3, dropout=0.2)
        
        self.ks_linear_embedding = LinearEmbedding(hidden_size, hidden_size)
        self.ks_pos_embedding = PositionalEncoding(hidden_size, max_len=100)
        
        self.pose_linear_embedding = LinearEmbedding(pose_size, hidden_size)
        self.pose_pos_embedding = PositionalEncoding(hidden_size, max_len=100)

        self.transformer = Transformer(in_size=hidden_size,
                                       hidden_size=hidden_size,
                                       num_hidden_layers=num_hidden_layers,
                                       num_attention_heads=num_attention_heads,
                                       intermediate_size=intermediate_size,
                                       is_compressed=is_compressed,
                                       cross_modal=True)
        
        self.out_fc = nn.Linear(hidden_size, out_size)
        # self.transformer = SoleFormer_Model(input_dim=hidden_size, d_model=768, nhead=4, num_encoder_layers=3, output_dim=64, is_seq2seq=True)
        # self.out = LinearEmbedding(hidden_size, out_size)

    def forward(self, ks, pose):
        # dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        
        ks = ks.permute(0, 2, 1)
        ks = self.TCN(ks) 
        ks = ks.permute(1, 0, 2)
        ks = self.ks_linear_embedding(ks)
        ks = self.ks_pos_embedding(ks)
        # print('ks shape', ks.shape)
        
        # pose [bs, ch, seq]
        pose = pose.permute(2, 0, 1)
        # pose [seq, bs, ch]
        pose = self.pose_linear_embedding(pose)
        pose = self.pose_pos_embedding(pose)
        # print('pose shape', pose.shape)

        x_data = {'x_a': pose, 'x_b': ks}
        # print('x permute', x.shape)
        
        x = self.transformer(x_data)
        x = self.out_fc(x)
        
        # print('x trans', x.shape)
        # x [seq, bs, ch]
        x = x.permute(1, 0, 2)
        # x [bs, seq, ch]
        x = x.permute(0, 2, 1)
        # x [bs, ch, seq]
        # print('x trans', x.shape)
        
        return x

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(out_channels),
#             )

#     def forward(self, x):
#         residual = self.downsample(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += residual
#         out = self.relu(out)
#         return out

# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
#         super(TransformerEncoderLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, src):
#         src2 = self.self_attn(src, src, src)[0]
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src

# class EMGFormer(nn.Module):
#     def __init__(self, in_channels=88, resnet_channels=512, n_res_blocks=4, n_transformer_layers=6, nhead=8, d_model=512, out_channels=96, seq_len=1024):
#         super(EMGFormer, self).__init__()
#         self.res_blocks = nn.Sequential(
#             *[ResidualBlock(in_channels if i == 0 else resnet_channels, resnet_channels) for i in range(n_res_blocks)]
#         )
#         self.linear = nn.Linear(resnet_channels, d_model)
#         self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))

#         self.transformer_layers = nn.ModuleList([
#             TransformerEncoderLayer(d_model, nhead) for _ in range(n_transformer_layers)
#         ])
        
#         self.final_linear = nn.Linear(d_model, out_channels)

#     def forward(self, x):
#         x = x.permute(0,2,1)
#         x = self.res_blocks(x)  # [bs, resnet_channels, seq_len]
#         x = x.permute(0, 2, 1)  # [bs, seq_len, resnet_channels]
#         x = self.linear(x) + self.positional_encoding[:, :x.size(1), :]  # [bs, seq_len, d_model]

#         for layer in self.transformer_layers:
#             x = layer(x)

#         x = x.permute(0, 2, 1)  # [bs, d_model, seq_len]
#         x = F.adaptive_avg_pool1d(x, 64)  # [bs, d_model, 64]
#         x = self.final_linear(x.permute(0, 2, 1))  # [bs, 64, out_channels]

#         return x