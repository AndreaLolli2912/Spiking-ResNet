"""
Module for SpikingDA Fusion Block: fuses multiple spiking streams
for SpikingResNet via temporal alignment and averaging.
"""
from torch import nn
import snntorch as snn
from snntorch import surrogate

class SpikingDABlock(nn.Module):
    """
    Data-Aware Spiking Block that fuses multiple spiking input streams.

    Configuration keys:
      - in_channels_list (list[int]): channels per input stream
      - out_channels (int): channels for fused output
      - downsample_strides (list[int]): temporal stride per input
      - n_steps (int): number of time steps
      - spike_grad: surrogate gradient for LIF neurons
      - dropout (float): dropout probability
    """
    def __init__(
        self,
        in_channels_list: list[int],
        out_channels: int,
        downsample_strides: list[int],
        config: dict
    ):
        super().__init__()
        # Hyperparameters from config
        self.n_steps = config.model.n_steps
        self.spike_grad = surrogate.atan(alpha=5)
        dp = config.model.dropout

        assert len(in_channels_list) == len(downsample_strides), \
            "in_channels_list and downsample_strides must match"
        self.num_inputs = len(in_channels_list)

        # Build per-input transforms, batch norms, and LIFs
        self.transforms = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lifs = nn.ModuleList()
        # Build a 1x1 conv for each input with its own stride
        for c_in, stride in zip(in_channels_list, downsample_strides):
            self.transforms.append(nn.Conv1d(c_in, out_channels, kernel_size=1, stride=stride))
            # time-step-specific batchnorm
            self.bns.append(nn.ModuleList([nn.BatchNorm1d(out_channels) for _ in range(self.n_steps)]))
            # one LIF per input stream
            self.lifs.append(snn.Leaky(beta=0.95, learn_beta=True, spike_grad=self.spike_grad))

        # Shared dropout after fusion
        self.dropout = nn.Dropout(dp)

    def forward(self, x):
        """
        Fuse multiple spiking sequences by averaging across inputs each time step.

        Args:
            x (tuple): (x_seq_list, mem_list) where
                x_seq_list (list[list[Tensor]]): length num_inputs, 
                    each a list of Tensors len n_steps
                mem_list (list[Tensor]): length num_inputs of initial membrane states
        Returns:
            out_seq (list[Tensor]): fused output per time step
            mem_list (list[Tensor]): updated membrane states
        """
        x_seq_list, mem_list = x
        out_seq: list = []

        for t in range(self.n_steps):
            sum_spk = 0
            for i, seq in enumerate(x_seq_list):
                x   = seq[t]
                x   = self.transforms[i](x)
                x   = self.bns[i][t](x)
                spk, new_mem = self.lifs[i](x, mem_list[i])
                sum_spk    += spk
                mem_list[i] = new_mem

            fused = sum_spk / len(x)
            fused = self.dropout(fused)
            out_seq.append(fused)

        return out_seq, mem_list
