"""
Define ConvLSTM model as forecasting baseline.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from operator import itemgetter


class ConvLSTMForecaster(nn.Module):
    def __init__(self, 
            in_channels: int,
            output_shape: tuple,
            channels: tuple,
            last_ts: bool = True,
            kernel_size: int = 3,
            last_relu: bool = True):
        super().__init__()

        self.last_ts = last_ts
        self.rnn = ConvLSTM(in_channels=in_channels, num_filter=channels[0], kernel_size=kernel_size,
                        patch_h=output_shape[1], patch_w=output_shape[2])
        self.out_layer1 = nn.Conv2d(channels[0], channels[1], kernel_size=1)
        self.out_layer2 = nn.Conv2d(channels[1], output_shape[0], 1)
        self.latlon = output_shape[1:]
        self.last_relu = last_relu
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        inputs = inputs.permute(1,0,2,3,4) # seq_first
        out, _ = self.rnn(inputs)
        
        if self.last_ts:
            out = out[-1]
        else:
            out = out.permute(1,0,2,3,4) # bsz_first
            bsz = len(out)
            out = out.contiguous().view(bsz, -1, *self.latlon) # use all time steps

        out = self.out_layer1(out)
        out = self.out_layer2(out)
        if self.last_relu:
            out = self.relu(out)
        return out


class ConvLSTM(nn.Module):
    """
    ConvLSTM based on https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py"""

    def __init__(self, in_channels: int, num_filter: int, kernel_size: int, patch_h: int, patch_w: int):
        super().__init__()
        self._state_height, self._state_width = patch_h, patch_w # patch dimensions after SpatialDownsampler
        self._conv = nn.Conv2d(in_channels=in_channels + num_filter,
                               out_channels=num_filter*4,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=1)

        # if using requires_grad flag, torch.save will not save parameters in deed although it may be updated every epoch.
        # Howerver, if you use declare an optimizer like Adam(model.parameters()),
        # parameters will not be updated forever.
        self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width))
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width))
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width))

        self._input_channel = in_channels
        self._num_filter = num_filter

    def init_hidden(self, inputs):
        c = inputs.new(size=(inputs.size(1), self._num_filter, self._state_height, self._state_width))
        h = inputs.new(size=(inputs.size(1), self._num_filter, self._state_height, self._state_width))
        return h, c

    # inputs and states should not be all none
    # inputs: S*B*C*H*W
    def forward(self, inputs):
        """
        Expected input shape [seq_len, bsz, channels, height, width]
        input shape (seq_len, bsz, 256, 64, 64)
        output[0] shape (seq_len, bsz, 384, 64, 64)
        """
        
        seq_len = len(inputs)
        self.hidden = self.init_hidden(inputs)
        h, c = self.hidden

        outputs = []
        for index in range(seq_len):
            # initial inputs
            if inputs is None:
                x = torch.zeros((h.size(0), self.in_channels, self._state_height, self._state_width), dtype=torch.float)
            else:
                x = inputs[index, ...]
            cat_x = torch.cat([x, h], dim=1)
            conv_x = self._conv(cat_x)
            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            # lstm equations
            i = torch.sigmoid(i+self.Wci*c)
            f = torch.sigmoid(f+self.Wcf*c)
            c = f*c + i*torch.tanh(tmp_c)
            o = torch.sigmoid(o+self.Wco*c)
            h = o*torch.tanh(c)

            outputs.append(h)
        outputs = torch.stack(outputs)

        return outputs, (h, c)