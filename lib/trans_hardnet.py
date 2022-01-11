import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from compare_models_speed.ATLAS_models.lib.transformer import TransformerModel


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        # print(x.shape, position_embeddings.shape, '？？？')
        return x + position_embeddings


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=kernel // 2, bias=False))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        # print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)

    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.n_layers = n_layers
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0  # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            layers_.append(ConvLayer(inch, outch))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        # print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or \
                    (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # print("upsample",in_channels, out_channels)

    def forward(self, x, skip, concat=True):
        out = F.interpolate(
            x,
            size=(skip.size(2), skip.size(3)),
            mode="bilinear",
            align_corners=False,
        )
        if concat:
            out = torch.cat([out, skip], 1)

        return out


#class hardnet(nn.Module):
class TransHarDNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=6):
        super(TransHarDNet, self).__init__()
        self.in_channels = n_channels
        self.embedding_dim = 512
        self.seq_length = int((512 // 8) ** 2)
        self.transformer = TransformerModel(
            dim=512,
            depth=4,
            heads=8,
            mlp_dim=4096,
        )
        self.conv_x = nn.Conv2d(
            320,
            self.embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # self.position_encoding = LearnedPositionalEncoding(
        #     self.seq_length, self.embedding_dim, self.seq_length)
        self.position_encoding = PositionalEncoding(
            self.embedding_dim)
        self.pre_head_ln = nn.LayerNorm(self.embedding_dim)
        self.conv1 = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1,
            padding=self._get_padding('VALID', (1, 1), ),
        )
        self.bn1 = nn.BatchNorm2d(self.embedding_dim)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=320,
            kernel_size=1,
            stride=1,
            padding=self._get_padding('VALID', (1, 1), ),
        )
        self.upsample = nn.Upsample( scale_factor=8, mode='bilinear', align_corners=False)

        first_ch = [16, 24, 32, 48]
        ch_list = [64, 96, 160, 224, 320]
        grmul = 1.7
        gr = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]

        blks = len(n_layers)
        self.shortcut_layers = []

        self.base = nn.ModuleList([])
        self.base.append( ConvLayer(in_channels=self.in_channels, out_channels=first_ch[0], kernel=3, stride=2))
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=3))
        self.base.append(ConvLayer(first_ch[1], first_ch[2], kernel=3, stride=1))      ## stride=1 for 256  stride=2 for 512
        self.base.append(ConvLayer(first_ch[2], first_ch[3], kernel=3))
        self.pe_dropout = nn.Dropout(p=0.1)

        skip_connection_channel_counts = []
        ch = first_ch[3]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)
            if i < blks - 1:
                self.shortcut_layers.append(len(self.base) - 1)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]

            if i < blks - 1:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))

        # self.base.append(ConvLayer(320, 320, kernel=1))
        # self.base.append(nn.Linear(20480, 8))
        # self.base.append(PositionalEncoding(320, dropout=0.2))

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks - 1
        self.n_blocks = n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])

        for i in range(n_blocks - 1, -1, -1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count // 2, kernel=1))
            cur_channels_count = cur_channels_count // 2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])

            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True)

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def encode(self, x):
        x = self.conv_x(x)
        # print(x.shape)
        x = x.permute(0, 2, 3, 1).contiguous()
        # print(x.shape, '---')
        x = x.view(x.size(0), -1, self.embedding_dim)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)
        # print(x.shape, type(x))

        return x, intmd_x

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            # int(self.img_dim / self.patch_dim),
            # int(self.img_dim / self.patch_dim),
            8,
            8,
            self.embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def decode(self, x):
        x = self._reshape_output(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        # x = self.upsample(x)
        return x

    def forward(self, x):
        skip_connections = []
        size_in = x.size()
        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)

        # print(x.shape, 'org')
        encoder_output, intmd_encoder_outputs = self.encode(x)
        decoder_output = self.decode(encoder_output)
        # print(decoder_output.shape, 'DE')

        out = decoder_output

        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)

        out = F.interpolate(
            out,
            # size=(size_in[2], size_in[3]),
            size=(224, 224),
            mode="bilinear",
            align_corners=False)
        return out


if __name__ == '__main__':
    model = TransHarDNet(1, 6).cuda()
    model.cuda()
    cuda0 = torch.device('cuda:0')
    # x = torch.rand((1, 1, 512, 512), device=cuda0)
    x = torch.rand((8, 1, 256, 256), device=cuda0)
    y = model(x)
    print(y.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print('Parameters:', total_params)
