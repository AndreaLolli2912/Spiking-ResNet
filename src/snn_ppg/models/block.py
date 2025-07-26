"""
Module for Spiking residual Block: one building block of SpikingResNet architecture.
"""
from torch import nn
import snntorch as snn
from snntorch import surrogate

class Block(nn.Module):
    """
    Spiking residual block consisting of two convolutional layers with spiking
    nonlinearities and an optional downsample projection for the skip connection.

    Args:
      in_channels (int): input channel count
      out_channels (int): output channel count
      stride (int): stride for first convolution (controls downsampling)
      config (dict): configuration dict containing:
        - n_steps (int): number of time steps
        - spike_grad: gradient surrogate for LIF neurons
        - dropout (float): dropout probability
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        config: dict
    ):
        super().__init__()
        # Hyperparameters from config
        self.n_steps = config.model.n_steps
        self.spike_grad = surrogate.atan(alpha=5)
        dp = config.model.dropout

        # --- main path ---
        self.bn1   = nn.ModuleList([nn.BatchNorm1d(in_channels)   for _ in range(self.n_steps)])
        self.lif1  = snn.Leaky(beta=0.95, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.conv1 = nn.Conv1d(in_channels, out_channels,kernel_size=3, stride=stride, padding=1)

        self.bn2   = nn.ModuleList([nn.BatchNorm1d(out_channels)  for _ in range(self.n_steps)])
        self.lif2  = snn.Leaky(beta=0.95, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # --- residual projection ---
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.downsample = None

        self.dropout = nn.Dropout(dp)

    def forward(self, x):
        """
        Forward pass for the spiking block.

        Args:
            x (tuple): Tuple of (x_seq, mem1, mem2):
                x_seq (list of Tensors): length n_steps; each Tensor [B, C, L]
                mem1 (Tensor): membrane state for first LIF
                mem2 (Tensor): membrane state for second LIF
        Returns:
            out_seq (list of Tensors): spike output per time step
            mem1, mem2 (Tensor): final membrane states
        """
        x_seq, mem1, mem2 = x
        out_seq = []

        for t in range(self.n_steps):
            x   = x_seq[t]
            res = x if self.downsample is None else self.downsample(x)

            out, mem1 = self.lif1(self.bn1[t](x), mem1)
            out       = self.conv1(out)

            out, mem2 = self.lif2(self.bn2[t](out), mem2)
            out       = self.conv2(out)

            # now out.shape == res.shape
            out = out + res
            out = self.dropout(out)
            out_seq.append(out)

        return out_seq, mem1, mem2
