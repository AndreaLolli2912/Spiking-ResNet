"""
Module for SpikingResNet: spiking ResNet architecture with residual and fusion blocks.
"""
# pylint: disable=R0902,R0914,R0913
import torch
import snntorch as snn
from snntorch import surrogate
from torch import nn

from .block import Block
from .dablock import SpikingDABlock

class Stem(nn.Module):
    

class SpikingResNet(nn.Module):
    """
    SpikingResNet combining residual Blocks and SpikingDABlocks with dual output heads.

    Configuration keys:
      - n_steps (int): number of time steps
      - initial_channels (int): channel width of stem
      - layers (list[int]): number of Blocks per stage
      - signal_channels (int): channels of input signals
      - num_classes (int): number of output classes
      - spike_grad: surrogate gradient for LIF neurons
      - encoding (bool): whether input x is list of length n_steps
      - dropout (float): dropout probability
    """
    def __init__(self, config: dict):
        super().__init__()
        
        # self.config = config
        # self.in_channels = 64
        n_steps = config.model.n_steps
        layers_cfg = config.model.layers
        signal_ch = config.model.signal_channels
        spike_grad = surrogate.atan(alpha=5)
        dp = config.model.dropout
        encoding = config.model.encoding

        # Stem
        beta0 = torch.rand(313) # I put this because I know the size of my data after stream
        self.conv0 = nn.Conv1d(signal_ch, config.model.initial_channels, kernel_size=7, stride=2, padding=3)
        self.bns0   = nn.ModuleList([nn.BatchNorm1d(config.model.initial_channels) for _ in range(n_steps)])
        self.lif0  = snn.Leaky(beta=beta0, learn_beta=True, learn_threshold=True, spike_grad=spike_grad)


        [config.model.initial_channels * 2 ** i for i in range(len(config.model.layers))]
        # Residual stages
        self.stage1 = self._make_stage(layers_cfg[0],  64, stride=1, config=config)
        self.stage2 = self._make_stage(layers_cfg[1], 128, stride=2, config=config)
        self.stage3 = self._make_stage(layers_cfg[2], 256, stride=2, config=config)
        self.stage4 = self._make_stage(layers_cfg[3], 512, stride=2, config=config)


        # DA fusion blocks with downsample_strides for each input
        self.da12   = SpikingDABlock([64, 64],                64,  downsample_strides=[1, 1],          config=config)
        self.da123  = SpikingDABlock([64, 64, 128],           128, downsample_strides=[2, 2, 1],       config=config)
        self.da1234 = SpikingDABlock([64, 64, 128, 256],      256, downsample_strides=[4, 4, 2, 1],    config=config)
        self.da_all = SpikingDABlock([64, 64, 128, 256, 512], 512, downsample_strides=[8, 8, 4, 2, 1], config=config)

        # Classifier head
        beta_out_1   = torch.rand(1)
        thr_out_1    = torch.rand(1)
        beta_out_2   = torch.rand(1)
        thr_out_2    = torch.rand(1)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout  = nn.Dropout(dp)

        self.fc_out_1   = nn.Linear(512, 1)
        self.lif_out_1 = snn.Leaky(beta=beta_out_1, threshold=thr_out_1, learn_beta=True, learn_threshold=True, spike_grad=spike_grad, reset_mechanism="none")

        self.fc_out_2   = nn.Linear(512, 1)
        self.lif_out_2 = snn.Leaky(beta=beta_out_2, threshold=thr_out_2, learn_beta=True, learn_threshold=True, spike_grad=spike_grad, reset_mechanism="none")

    def forward(self, x):
        # helper to reset membrane states
        def reset(lif):
            return lif.reset_mem()

        # reset & store
        mem_stem    = reset(self.lif0)
        mem_stage1, mem_stage2, mem_stage3, mem_stage4 = [
            [[reset(blk.lif1), reset(blk.lif2)] for blk in stage]
            for stage in (self.stage1, self.stage2, self.stage3, self.stage4)
        ]
        mem_da12   = [reset(l) for l in self.da12.lifs]
        mem_da123  = [reset(l) for l in self.da123.lifs]
        mem_da1234 = [reset(l) for l in self.da1234.lifs]
        mem_da_all = [reset(l) for l in self.da_all.lifs]

        mem_cls_1  = reset(self.lif_out_1)
        mem_cls_2  = reset(self.lif_out_2)

        # f0 / stem
        stem_seq  = []

        for t in range(self.n_steps):
            if self.encoding:
                x_step = x[t]
            else:
                x_step = x

            cur_stem = self.bns0[t](self.conv0(x_step))
            spk, mem_stem = self.lif0(cur_stem, mem_stem)
            stem_seq.append(spk)

        # stage 1
        seq = stem_seq
        for idx, blk in enumerate(self.stage1):
            m1, m2 = mem_stage1[idx]
            seq, m1, m2 = blk([seq, m1, m2])
            mem_stage1[idx] = [m1, m2]
        out_seq1_da, mem_da12 = self.da12(([stem_seq, seq], mem_da12))
        out_seq1p = [f0 + d for f0, d in zip(stem_seq, out_seq1_da)]

        # stage 2
        seq = out_seq1p
        for idx, blk in enumerate(self.stage2):
            m1, m2 = mem_stage2[idx]
            seq, m1, m2 = blk([seq, m1, m2])
            mem_stage2[idx] = [m1, m2]

        out_seq2_da, mem_da123 = self.da123(([stem_seq, out_seq1p, seq], mem_da123))
        out_seq2p = [f2 + d for f2, d in zip(seq, out_seq2_da)]

        # stage 3
        seq = out_seq2p
        for idx, blk in enumerate(self.stage3):
            m1, m2 = mem_stage3[idx]
            seq, m1, m2 = blk([seq, m1, m2])
            mem_stage3[idx] = [m1, m2]
        out_seq3_da, mem_da1234 = self.da1234(([
            stem_seq, out_seq1p, out_seq2p, seq
        ], mem_da1234))
        out_seq3p = [f3 + d for f3, d in zip(seq, out_seq3_da)]

        # stage 4
        seq = out_seq3p
        for idx, blk in enumerate(self.stage4):
            m1, m2 = mem_stage4[idx]
            seq, m1, m2 = blk([seq, m1, m2])
            mem_stage4[idx] = [m1, m2]
        out_seq4_da, mem_da_all = self.da_all(([
            stem_seq, out_seq1p, out_seq2p, out_seq3p, seq
        ], mem_da_all))
        out_seq4p = [f4 + d for f4, d in zip(seq, out_seq4_da)]

        # classifier head
        mem_seq_1 = []
        mem_seq_2 = []
        for t in range(self.n_steps):

            pooled = self.avg_pool(out_seq4p[t]).squeeze(-1)
            pooled = self.dropout(pooled)

            cur_cls_1 = self.fc_out_1(pooled)
            spk_1, mem_cls_1 = self.lif_out_1(cur_cls_1, mem_cls_1)

            cur_cls_2 = self.fc_out_2(pooled)
            spk_2, mem_cls_2 = self.lif_out_2(cur_cls_2, mem_cls_2)

            mem_seq_1.append(mem_cls_1)
            mem_seq_2.append(mem_cls_2)

        mem_seq_1 = torch.stack(mem_seq_1, dim=0)
        mem_seq_2 = torch.stack(mem_seq_2, dim=0)

        mem_seq = torch.cat([mem_seq_1, mem_seq_2], dim=2)

        # return the list of final membrane states across time
        return mem_seq

    def _make_stage(self, num_blocks, out_channels, stride, config):
        blocks = []
        for i in range(num_blocks):
            blk_stride = stride if i == 0 else 1

            blocks.append(Block(self.in_channels, out_channels, blk_stride, config))

            self.in_channels = out_channels

        return nn.ModuleList(blocks)
