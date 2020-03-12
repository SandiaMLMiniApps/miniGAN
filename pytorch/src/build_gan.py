##########################################################################
# ************************************************************************
#
#               miniGAN : GAN Proxy Application
#                 Copyright 2019 Sandia Corporation
#
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the Corporation nor the names of the
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Questions? Contact J. Austin Ellis (johelli@sandia.gov)
#
# ************************************************************************
##########################################################################

from __future__ import print_function, division

import sys, os
from os import path
import warnings
import timeit

import numpy as np
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torch.utils.data.distributed
import horovod.torch as hvd


# New Pytorch CPP kk conv ops
#kk_conv2d_module = tf.load_op_library('./build2/libkk_conv2d.so')
#kk_conv3d_module = tf.load_op_library('./build3/libkk_conv3d.so')

# New Pytorch Layers
#from kk_conv2d import KK_Conv2D_Layer
#from kk_conv3d import KK_Conv3D_Layer

# Pytorch kk conv grads
#import kk_conv2d_grad
#import kk_conv3d_grad


# 2D Generator Network
class Generator2d(nn.Module):
    def __init__(self, minigan_args):
        super(Generator2d, self).__init__()

        self.args = minigan_args
        self.ngpu = self.args.num_gpus

        self.gen2d = nn.Sequential()

        i_channels = self.args.gen_noise
        o_channels = self.args.gen_filters * (2 ** self.args.gen_layers)

        if (self.args.kk_mode):

            if (hvd.rank() == 0):
                print("\nRequested KokkosKernels mode, but it is currently unavailable. " \
                        "Using default Pytorch layers.\n")
            self.gen2d.add_module("kk_conv_trans_%d" % (0), \
                    nn.ConvTranspose2d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.gen_kern_size, \
                    stride=1, \
                    padding=0, \
                    output_padding=0, \
                    groups=1, \
                    bias=False, \
                    dilation=1))
        else:
            self.gen2d.add_module("conv_trans_%d" % (0), \
                    nn.ConvTranspose2d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.gen_kern_size, \
                    stride=1, \
                    padding=0, \
                    output_padding=0, \
                    groups=1, \
                    bias=False, \
                    dilation=1))
        # Batch normalization
        self.gen2d.add_module("batch_norm_%d" % (0), \
                nn.BatchNorm2d( \
                num_features=o_channels, \
                eps=self.args.bn_eps, \
                momentum=self.args.bn_mom, \
                affine=True, \
                track_running_stats=True))
        # Leaky ReLU activation
        self.gen2d.add_module("relu_%d" % (0), \
                nn.ReLU(True))

        for l in range(self.args.gen_layers):

            i_channels = o_channels
            o_channels = i_channels // 2

            # KK transposed convolutions (currently unavailable)
            if (self.args.kk_mode):
                # 2d Transposed Convolution
                self.gen2d.add_module("kk_conv_trans_%d" % (l+1), \
                    nn.ConvTranspose2d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.gen_kern_size, \
                    stride=self.args.gen_kern_stride, \
                    padding=self.args.gen_padding, \
                    output_padding=0, \
                    groups=1, \
                    bias=False, \
                    dilation=1))

            # PyTorch transposed convolutions
            else:
                # 2d Transposed Convolution
                self.gen2d.add_module("conv_trans_%d" % (l+1), \
                    nn.ConvTranspose2d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.gen_kern_size, \
                    stride=self.args.gen_kern_stride, \
                    padding=self.args.gen_padding, \
                    output_padding=0, \
                    groups=1, \
                    bias=False, \
                    dilation=1))
            # Batch normalization
            self.gen2d.add_module("batch_norm_%d" % (l+1), \
                    nn.BatchNorm2d( \
                    num_features=o_channels, \
                    eps=self.args.bn_eps, \
                    momentum=self.args.bn_mom, \
                    affine=True, \
                    track_running_stats=True))
            # Leaky ReLU activation
            self.gen2d.add_module("relu_%d" % (l+1), \
                    nn.ReLU(True))

        # Last layer outputs num channels in dataset
        # args.gen_filters = o_channels after finshing for-loop
        i_channels = self.args.gen_filters
        o_channels = self.args.num_channels

        if (self.args.kk_mode):
            self.gen2d.add_module("last_kk_conv_trans", \
                    nn.ConvTranspose2d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.gen_kern_size, \
                    stride=self.args.gen_kern_stride, \
                    padding=self.args.gen_padding, \
                    output_padding=0, \
                    groups=1, \
                    bias=False, \
                    dilation=1))
        else:
            self.gen2d.add_module("last_conv_trans", \
                    nn.ConvTranspose2d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.gen_kern_size, \
                    stride=self.args.gen_kern_stride, \
                    padding=self.args.gen_padding, \
                    output_padding=0, \
                    groups=1, \
                    bias=False, \
                    dilation=1))

        self.gen2d.add_module("tanh", \
                nn.Tanh())

    def forward(self, input):
        return self.gen2d(input)

# 2D Discriminator Network
class Discriminator2d(nn.Module):
    def __init__(self, minigan_args):
        super(Discriminator2d, self).__init__()

        self.args = minigan_args
        self.ngpu = self.args.num_gpus

        self.disc2d = nn.Sequential()

        i_channels = self.args.num_channels
        o_channels = self.args.disc_filters

        # KK convolutions (currently unavailable)
        if (self.args.kk_mode):
            if (hvd.rank() == 0):
                print("\nRequested KokkosKernels mode, but it is currently unavailable. ", \
                        "Using default Pytorch layers.\n")
            # 2d Convolution
            self.disc2d.add_module("kk_conv_%d" % (0), \
                    nn.Conv2d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.disc_kern_size, \
                    stride=self.args.disc_kern_stride, \
                    padding=self.args.disc_padding, \
                    dilation=1, \
                    groups=1, \
                    bias=False))
        # PyTorch convolutions
        else:
            # 2d Convolution
            self.disc2d.add_module("conv_%d" % (0), \
                    nn.Conv2d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.disc_kern_size, \
                    stride=self.args.disc_kern_stride, \
                    padding=self.args.disc_padding, \
                    dilation=1, \
                    groups=1, \
                    bias=False))

        # Leaky ReLU activation
        self.disc2d.add_module("relu_%d" % (0), \
            nn.LeakyReLU(self.args.leaky_relu, inplace=True))

        for l in range(self.args.disc_layers):

            i_channels = o_channels
            o_channels = i_channels * 2

            # KK convolutions (currently unavailable)
            if (self.args.kk_mode):
                # 2d Convolution
                self.disc2d.add_module("kk_conv_%d" % (l+1), \
                        nn.Conv2d( \
                        in_channels=i_channels, \
                        out_channels=o_channels, \
                        kernel_size=self.args.disc_kern_size, \
                        stride=self.args.disc_kern_stride, \
                        padding=self.args.disc_padding, \
                        dilation=1, \
                        groups=1, \
                        bias=False))
            # PyTorch convolutions
            else:
                # 2d Convolution
                self.disc2d.add_module("conv_%d" % (l+1), \
                        nn.Conv2d( \
                        in_channels=i_channels, \
                        out_channels=o_channels, \
                        kernel_size=self.args.disc_kern_size, \
                        stride=self.args.disc_kern_stride, \
                        padding=self.args.disc_padding, \
                        dilation=1, \
                        groups=1, \
                        bias=False))

            # Batch Normalization
            self.disc2d.add_module("batch_norm_%d" % (l+1), \
                    nn.BatchNorm2d( \
                    num_features=o_channels, \
                    eps=1e-05, \
                    momentum=0.1, \
                    affine=True, \
                    track_running_stats=True))
            # Leaky ReLU activation
            self.disc2d.add_module("relu_%d" % (l+1), \
                    nn.LeakyReLU(self.args.leaky_relu, inplace=True))

        # Last layer outputs real/fake label, so 1 channel
        # args.disc_filters * 2 ** disc_layers = o_channels after finshing for-loop
        i_channels = self.args.disc_filters * (2 ** self.args.disc_layers)
        o_channels = 1

        # KK convolutions (currently unavailable)
        if (self.args.kk_mode):
            # 2d Convolution
            self.disc2d.add_module("last_kk_conv", \
                    nn.Conv2d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.disc_kern_size, \
                    stride=1, \
                    padding=0, \
                    dilation=1, \
                    groups=1, \
                    bias=False))
        # PyTorch convolutions
        else:
            # 2d Convolution
            self.disc2d.add_module("last_conv", \
                    nn.Conv2d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.disc_kern_size, \
                    stride=1, \
                    padding=0, \
                    dilation=1, \
                    groups=1, \
                    bias=False))

        self.disc2d.add_module("sigmoid", \
                nn.Sigmoid())

    def forward(self, input):
        return self.disc2d(input)

# 3D Generator Network
class Generator3d(nn.Module):
    def __init__(self, minigan_args):
        super(Generator3d, self).__init__()

        self.args = minigan_args
        self.ngpu = self.args.num_gpus

        self.gen3d = nn.Sequential()

        i_channels = self.args.gen_noise
        o_channels = self.args.gen_filters * (2 ** self.args.gen_layers)

        if (self.args.kk_mode):
            if (hvd.rank() == 0):
                print("\nRequested KokkosKernels mode, but it is currently unavailable. ", \
                        "Using default Pytorch layers.\n")
            self.gen3d.add_module("kk_conv_trans_%d" % (0), \
                    nn.ConvTranspose3d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.gen_kern_size, \
                    stride=1, \
                    padding=0, \
                    output_padding=0, \
                    groups=1, \
                    bias=False, \
                    dilation=1))
        else:
            self.gen3d.add_module("conv_trans_%d" % (0), \
                    nn.ConvTranspose3d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.gen_kern_size, \
                    stride=1, \
                    padding=0, \
                    output_padding=0, \
                    groups=1, \
                    bias=False, \
                    dilation=1))
        # Batch normalization
        self.gen3d.add_module("batch_norm_%d" % (0), \
                nn.BatchNorm3d( \
                num_features=o_channels, \
                eps=self.args.bn_eps, \
                momentum=self.args.bn_mom, \
                affine=True, \
                track_running_stats=True))
        # Leaky ReLU activation
        self.gen3d.add_module("relu_%d" % (0), \
                nn.ReLU(True))

        for l in range(self.args.gen_layers):

            i_channels = o_channels
            o_channels = i_channels // 2

            # KK transposed convolutions (currently unavailable)
            if (self.args.kk_mode):
                # 3d Transposed Convolution
                self.gen3d.add_module("kk_conv_trans_%d" % (l+1), \
                    nn.ConvTranspose3d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.gen_kern_size, \
                    stride=self.args.gen_kern_stride, \
                    padding=self.args.gen_padding, \
                    output_padding=0, \
                    groups=1, \
                    bias=False, \
                    dilation=1))

            # PyTorch transposed convolutions
            else:
                # 3d Transposed Convolution
                self.gen3d.add_module("conv_trans_%d" % (l+1), \
                    nn.ConvTranspose3d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.gen_kern_size, \
                    stride=self.args.gen_kern_stride, \
                    padding=self.args.gen_padding, \
                    output_padding=0, \
                    groups=1, \
                    bias=False, \
                    dilation=1))
            # Batch normalization
            self.gen3d.add_module("batch_norm_%d" % (l+1), \
                    nn.BatchNorm3d( \
                    num_features=o_channels, \
                    eps=self.args.bn_eps, \
                    momentum=self.args.bn_mom, \
                    affine=True, \
                    track_running_stats=True))
            # Leaky ReLU activation
            self.gen3d.add_module("relu_%d" % (l+1), \
                    nn.ReLU(True))

        # Last layer outputs num channels in dataset
        # args.gen_filters = o_channels after finshing for-loop
        i_channels = self.args.gen_filters
        o_channels = self.args.num_channels

        if (self.args.kk_mode):
            self.gen3d.add_module("last_kk_conv_trans", \
                    nn.ConvTranspose3d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.gen_kern_size, \
                    stride=self.args.gen_kern_stride, \
                    padding=self.args.gen_padding, \
                    output_padding=0, \
                    groups=1, \
                    bias=False, \
                    dilation=1))
        else:
            self.gen3d.add_module("last_conv_trans", \
                    nn.ConvTranspose3d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.gen_kern_size, \
                    stride=self.args.gen_kern_stride, \
                    padding=self.args.gen_padding, \
                    output_padding=0, \
                    groups=1, \
                    bias=False, \
                    dilation=1))

        self.gen3d.add_module("tanh", \
                nn.Tanh())

    def forward(self, input):
        return self.gen3d(input)

# 3D Discriminator Network
class Discriminator3d(nn.Module):
    def __init__(self, minigan_args):
        super(Discriminator3d, self).__init__()

        self.args = minigan_args
        self.ngpu = self.args.num_gpus

        self.disc3d = nn.Sequential()

        i_channels = self.args.num_channels
        o_channels = self.args.disc_filters

        # KK convolutions (currently unavailable)
        if (self.args.kk_mode):
            if (hvd.rank() == 0):
                print("\nRequested KokkosKernels mode, but it is currently unavailable. ", \
                        "Using default Pytorch layers.\n")
            # 3d Convolution
            self.disc3d.add_module("kk_conv_%d" % (0), \
                    nn.Conv3d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.disc_kern_size, \
                    stride=self.args.disc_kern_stride, \
                    padding=self.args.disc_padding, \
                    dilation=1, \
                    groups=1, \
                    bias=False))
        # PyTorch convolutions
        else:
            # 3d Convolution
            self.disc3d.add_module("conv_%d" % (0), \
                    nn.Conv3d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.disc_kern_size, \
                    stride=self.args.disc_kern_stride, \
                    padding=self.args.disc_padding, \
                    dilation=1, \
                    groups=1, \
                    bias=False))

        # Leaky ReLU activation
        self.disc3d.add_module("relu_%d" % (0), \
            nn.LeakyReLU(self.args.leaky_relu, inplace=True))

        for l in range(self.args.disc_layers):

            i_channels = o_channels
            o_channels = i_channels * 2

            # KK convolutions (currently unavailable)
            if (self.args.kk_mode):
                # 3d Convolution
                self.disc3d.add_module("kk_conv_%d" % (l+1), \
                        nn.Conv3d( \
                        in_channels=i_channels, \
                        out_channels=o_channels, \
                        kernel_size=self.args.disc_kern_size, \
                        stride=self.args.disc_kern_stride, \
                        padding=self.args.disc_padding, \
                        dilation=1, \
                        groups=1, \
                        bias=False))
            # PyTorch convolutions
            else:
                # 3d Convolution
                self.disc3d.add_module("conv_%d" % (l+1), \
                        nn.Conv3d( \
                        in_channels=i_channels, \
                        out_channels=o_channels, \
                        kernel_size=self.args.disc_kern_size, \
                        stride=self.args.disc_kern_stride, \
                        padding=self.args.disc_padding, \
                        dilation=1, \
                        groups=1, \
                        bias=False))

            # Batch Normalization
            self.disc3d.add_module("batch_norm_%d" % (l+1), \
                    nn.BatchNorm3d( \
                    num_features=o_channels, \
                    eps=1e-05, \
                    momentum=0.1, \
                    affine=True, \
                    track_running_stats=True))
            # Leaky ReLU activation
            self.disc3d.add_module("relu_%d" % (l+1), \
                    nn.LeakyReLU(self.args.leaky_relu, inplace=True))

        # Last layer outputs real/fake label, so 1 channel
        # args.disc_filters * 2 ** disc_layers = o_channels after finshing for-loop
        i_channels = self.args.disc_filters * (2 ** self.args.disc_layers)
        o_channels = 1

        # KK convolutions (currently unavailable)
        if (self.args.kk_mode):
            # 3d Convolution
            self.disc3d.add_module("last_kk_conv", \
                    nn.Conv3d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.disc_kern_size, \
                    stride=1, \
                    padding=0, \
                    dilation=1, \
                    groups=1, \
                    bias=False))
        # PyTorch convolutions
        else:
            # 3d Convolution
            self.disc3d.add_module("last_conv", \
                    nn.Conv3d( \
                    in_channels=i_channels, \
                    out_channels=o_channels, \
                    kernel_size=self.args.disc_kern_size, \
                    stride=1, \
                    padding=0, \
                    dilation=1, \
                    groups=1, \
                    bias=False))

        self.disc3d.add_module("sigmoid", \
                nn.Sigmoid())

    def forward(self, input):
        return self.disc3d(input)



# weight initialization for discriminator and generator
def weight_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)

#
# Generative Adversarial Network proxy application using Kokkos Kernels
#
class miniGAN:

    # Constructor
    def __init__(self, minigan_args):
        if (hvd.rank() == 0):
            print('Hello minigan init\n')

        self.minigan_args = minigan_args

        self.minigan_args.output_interimages_dir = \
                self.minigan_args.output_dir + "/inter_%ss" % (self.minigan_args.dataset)

        # Load datasets
        self.load_data()

        if (hvd.rank() == 0):
            print('\n---LOAD DATA DONE---\n')

        # 2D miniGAN
        if (self.minigan_args.dim_mode == 2):
            self.generator =     Generator2d(minigan_args)
            self.discriminator = Discriminator2d(minigan_args)
        # 3D miniGAN
        elif (self.minigan_args.dim_mode == 3):
            self.generator =     Generator3d(minigan_args)
            self.discriminator = Discriminator3d(minigan_args)
        else:
            raise ValueError('\'dim_mode\' must be {2} or {3}.')

        self.generator = self.generator.float()
        self.discriminator = self.discriminator.float()

        self.generator.apply(weight_init)
        self.discriminator.apply(weight_init)


        if (hvd.rank() == 0):
            print(self.generator)
            print(self.discriminator)

        # Metrics and Loss
        self.loss_fn = nn.BCELoss()

        self.real_label = 1.0 - self.minigan_args.soft_label
        self.fake_label = self.minigan_args.soft_label

        self.gen_optim = optim.Adam( \
                self.generator.parameters(), \
                lr=self.minigan_args.gen_lr, \
                betas=(self.minigan_args.gen_beta1, .999))
        self.disc_optim = optim.Adam( \
                self.discriminator.parameters(), \
                lr=self.minigan_args.disc_lr, \
                betas=(self.minigan_args.disc_beta1, .999))

        if self.minigan_args.cuda:
            self.generator.cuda()
            self.discriminator.cuda()

        hvd.broadcast_parameters(self.generator.state_dict(), root_rank=0)
        hvd.broadcast_parameters(self.discriminator.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.gen_optim, root_rank=0)
        hvd.broadcast_optimizer_state(self.disc_optim, root_rank=0)

        self.compression = \
                hvd.Compression.fp16 if self.minigan_args.fp16_allreduce else hvd.Compression.none

        self.gen_optim = hvd.DistributedOptimizer(self.gen_optim, \
                named_parameters=self.generator.named_parameters(), \
                compression = self.compression)

        self.disc_optim = hvd.DistributedOptimizer(self.disc_optim, \
                named_parameters=self.discriminator.named_parameters(), \
                compression = self.compression)

#        self.tb_disc.set_model(self.discriminator)
#        self.tb_comb_gan.set_model(self.combined_gan)

        if (self.minigan_args.profile):
            self.profile_layers(self.minigan_args.prof_steps, self.prof_images)

        if (hvd.rank() == 0):
            print('\n---NETWORK SETUP DONE---\n')


    #########################
    # Load miniGAN Training Data
    #########################
    def load_data(self):
        if (hvd.rank() == 0):
            print('\nHello load data\n')
            print('Loading dataset ' + self.minigan_args.dataset + '\n')

        # This has been known to cause issues if running on Summit with ddlrun
        # Reduce data_workers to 0 and pin_memory to False
        kwargs = {'num_workers': self.minigan_args.data_workers, 'pin_memory': True} if self.minigan_args.cuda else {}

        # Random Dataset
        if (self.minigan_args.dataset == "random"):
            if (self.minigan_args.dim_mode == 2):

                train_tensor_x = torch.randn( [\
                        self.minigan_args.num_images, \
                        self.minigan_args.num_channels, \
                        self.minigan_args.image_dim, \
                        self.minigan_args.image_dim], dtype=float)

            elif (self.minigan_args.dim_mode == 3):

                train_tensor_x = torch.randn([ \
                        self.minigan_args.num_images, \
                        self.minigan_args.num_channels, \
                        self.minigan_args.image_dim, \
                        self.minigan_args.image_dim, \
                        self.minigan_args.image_dim], dtype=float)

            else:
               raise ValueError('\'dim_mode\' must be {2} or {3}.')


            train_tensor_y = torch.ones([self.minigan_args.num_images, 1], dtype=float)

            train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)

            self.train_sampler = torch.utils.data.distributed.DistributedSampler( \
                    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

            self.train_loader = torch.utils.data.DataLoader( \
                    train_dataset, batch_size=self.minigan_args.batch_size, \
                    sampler=self.train_sampler, **kwargs)

        # Bird Dataset
        elif (self.minigan_args.dataset == "bird"):
            if (self.minigan_args.dim_mode == 2):

                bird_filename = self.minigan_args.data_dir + "/" + \
                        "minigan_bird_%dimgs_%dchnls_%dx%dpx.npy" % \
                        (self.minigan_args.num_images, \
                         self.minigan_args.num_channels, \
                         self.minigan_args.image_dim, \
                         self.minigan_args.image_dim)

                if (not path.exists(bird_filename)):
                    if (hvd.rank() == 0):
                        print("Can not find dataset for file: " + bird_filename + "\nEnsure it exists!")
                    exit(0)

            elif (self.minigan_args.dim_mode == 3):
                bird_filename = self.minigan_args.data_dir + "/" + \
                        "minigan_bird_%dimgs_%dchnls_%dx%dx%dpx.npy" % \
                        (self.minigan_args.num_images, \
                         self.minigan_args.num_channels, \
                         self.minigan_args.image_dim, \
                         self.minigan_args.image_dim, \
                         self.minigan_args.image_dim)

                if (not path.exists(bird_filename)):
                    if (hvd.rank() == 0):
                        print("Can not find dataset for file: " + bird_filename + "\nEnsure it exists!")
                    exit(0)


            else:
               raise ValueError('\'dim_mode\' must be {2} or {3}.')

            train_tensor_x = torch.tensor(np.load(bird_filename))
            train_tensor_y = torch.ones([self.minigan_args.num_images, 1], dtype=float)

            train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)

            self.train_sampler = torch.utils.data.distributed.DistributedSampler( \
                    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

            self.train_loader = torch.utils.data.DataLoader( \
                    train_dataset, batch_size=self.minigan_args.batch_size, \
                    sampler=self.train_sampler, **kwargs)

        else:
            raise ValueError('\'data\' must be {random or bird}.')


    # Profile layers of Discriminator and Generator
    def profile_layers(self, dummy_runs, p_samples):
        if (hvd.rank() == 0):
            print('Hello profiler!\n')

        ### Profile Discriminator ###
        if (hvd.rank() == 0):
            print('\n\n---Profiling Discriminator---\n\n')

        ### Profile GAN ###
        if (hvd.rank() == 0):
            print('\n\n---Profiling GAN---\n\n')

        if (hvd.rank() == 0):
            print("In development")

        sys.exit("\n\n---Done Profiling. Exiting---\n\n")


    # Train GAN for one epoch
    def run_epoch(self, current_epoch, num_batches):

        # Disc only timers
        disc_real_forward_time = 0.0
        disc_fake_forward_time = 0.0

        # Disc(Gen(noise)) timers
        gen_disc_fake_forward_time = 0.0
        gen_fake_forward_time = 0.0

        # Gen/Disc autograd timers
        disc_real_backward_time = 0.0
        disc_fake_backward_time = 0.0
        gen_fake_backward_time = 0.0

        # Gen/Disc weight update timers
        disc_apply_time = 0.0
        gen_apply_time = 0.0

        for batch_idx, (data, target) in enumerate(self.train_loader):

            batch_iteration = current_epoch * num_batches + batch_idx
            total_batches = self.minigan_args.epochs * num_batches

            current_samples = data.shape[0]

            #################################
            # Disc Update, Real Data
            #################################

            # Ensure gradients are recorded for the discriminator
            for p in self.discriminator.parameters():
                p.requires_grad = True
            self.discriminator.zero_grad()

            # Get data and labels to device
            real_device = data.to(self.minigan_args.device)
            b_size = real_device.size(0)
            label = torch.full((b_size,), self.real_label, device=self.minigan_args.device)

            # Forward discriminator pass for real data
            tic = timeit.default_timer()
            disc_real_output = self.discriminator(real_device.float()).view(-1)
            toc = timeit.default_timer()
            disc_real_forward_time += toc - tic

            # Calculate loss for real data
            disc_real_err = self.loss_fn(disc_real_output, label)

            # Calculate disc gradients for real images
            tic = timeit.default_timer()
            disc_real_err.backward()
            toc = timeit.default_timer()
            disc_real_backward_time += toc - tic

            # Average discriminator guesses for real data (should go to 1.0)
            disc_real_x = disc_real_output.mean().item()

            # Update discriminator
            tic = timeit.default_timer()
            self.disc_optim.step()
            toc = timeit.default_timer()
            disc_apply_time += toc - tic


            #################################
            # Disc Update, Fake Data
            #################################

            # Generate batch of latent vectors (std normal distribution)
            # 2D: (batchsize, nz, 1, 1)
            if(self.minigan_args.dim_mode == 2):
                noise = torch.randn([ \
                    current_samples, \
                    self.minigan_args.gen_noise, 1, 1], \
                    dtype=float, \
                    device=self.minigan_args.device)
            # 3D: (batchsize, nz, 1, 1)
            elif(self.minigan_args.dim_mode == 3):
                noise = torch.randn([ \
                    current_samples, \
                    self.minigan_args.gen_noise, 1, 1, 1], \
                    dtype=float, \
                    device=self.minigan_args.device)

            # Generate fake images and labels
            tic = timeit.default_timer()
            fake_device = self.generator(noise.float())
            toc = timeit.default_timer()
            gen_fake_forward_time += toc - tic

            label.fill_(self.fake_label)

            # Classify fake images with discriminator
            tic = timeit.default_timer()
            disc_fake_output = self.discriminator(fake_device.detach()).view(-1)
            toc = timeit.default_timer()
            disc_fake_forward_time += toc - tic

            # Calculate loss for fake data
            disc_fake_err = self.loss_fn(disc_fake_output, label)

            # Calculate disc gradients for fake images
            tic = timeit.default_timer()
            disc_fake_err.backward()
            toc = timeit.default_timer()
            disc_fake_backward_time += toc - tic

            # Calculate discriminator gradient norm (real and fake combined)
            d_grad_norm = 0.0
            for p in self.discriminator.parameters():
                param_norm = p.grad.data.norm(2)
                d_grad_norm += param_norm.item() ** 2
            d_grad_norm = d_grad_norm ** (1. / 2)

            # Average discriminator guesses for fake data (should go to 0.0)
            disc_fake_x = disc_fake_output.mean().item()

            # Combine errors for real and fake data
            disc_err = disc_real_err + disc_fake_err

            # Update discriminator
            tic = timeit.default_timer()
            self.disc_optim.step()
            toc = timeit.default_timer()
            disc_apply_time += toc - tic

            #################################
            # Gen Update, Fake Data
            #################################

            # Do not record gradients for the discriminator
            for p in self.discriminator.parameters():
                p.requires_grad = False
            self.generator.zero_grad()

            label.fill_(self.real_label)

            tic = timeit.default_timer()
            gen_output = self.discriminator(fake_device).view(-1)
            toc = timeit.default_timer()
            gen_disc_fake_forward_time += toc - tic

            # Calculate loss for fake data with generator
            gen_err = self.loss_fn(gen_output, label)

            # Calculate gen gradients for fake images
            tic = timeit.default_timer()
            gen_err.backward()
            toc = timeit.default_timer()
            gen_fake_backward_time += toc - tic

            # Calculate generator gradient norm
            g_grad_norm = 0.0
            for p in self.generator.parameters():
                param_norm = p.grad.data.norm(2)
                g_grad_norm += param_norm.item() ** 2
            g_grad_norm = g_grad_norm ** (1. / 2)

            # Average discriminator guesses for fake data (should go to 0.0)
            gan_fake_x = gen_output.mean().item()

            # Update generator
            tic = timeit.default_timer()
            self.gen_optim.step()
            toc = timeit.default_timer()
            gen_apply_time += toc - tic

            if (batch_idx % self.minigan_args.log_interval == 0 and hvd.rank() == 0):
                print("\nEpoch %d, Batch %d of %d, \
                       Cumulative Batch %d of %d!!!" % (current_epoch, batch_idx, \
                                                        num_batches, batch_iteration, \
                                                        total_batches))
                print("Discriminator loss: %f"%(disc_err.item()))
                print("Generator loss: %f"%(gen_err.item()))
                print("Disc Gradient norm: %f"%(d_grad_norm))
                print("Gen Gradient norm: %f"%(g_grad_norm))
                print("Average Disc real guesses: %f" % disc_real_x)
                print("Average Disc fake guesses: %f" % disc_fake_x)
                print("Gen/Disc fake guesses: %f" % gan_fake_x)

        forward_time = \
                disc_real_forward_time + \
                disc_fake_forward_time + \
                gen_disc_fake_forward_time + \
                gen_fake_forward_time

        backprop_time = \
                disc_real_backward_time + \
                disc_fake_backward_time + \
                gen_fake_backward_time

        apply_time = disc_apply_time + gen_apply_time

        tot_time = forward_time + backprop_time + apply_time

        if (hvd.rank() == 0):
            print("\n\nProfile for Epoch %d \
                   \nForward Time: \t\t\t%4.4fs, \
                   \nBackprop Time: \t\t\t%4.4fs, \
                   \nApply Time: \t\t\t%4.4fs" % \
                    (current_epoch, forward_time, backprop_time, apply_time))
            print("\nForward Percent: \t\t%2.2f%%, \
                   \nBackprop Percent: \t\t%2.2f%%, \
                   \nApply Percent: \t\t\t%2.2f%%" % \
                    (100 * forward_time / tot_time, \
                     100 * backprop_time / tot_time, \
                     100 * apply_time / tot_time))

        if (self.minigan_args.dim_mode == 2):
            if (current_epoch == 0):
                vutils.save_image(real_device, \
                        self.minigan_args.output_interimages_dir + "/real_samples.png", normalize=True)

            vutils.save_image(fake_device, self.minigan_args.output_interimages_dir + \
                    "/fake_samples_epoch%d.png" % current_epoch, normalize=True)
        else:
            if (current_epoch == 0):
                vutils.save_image(real_device[:,:,:,:,0], \
                        self.minigan_args.output_interimages_dir + "/real_samples.png", normalize=True)

            vutils.save_image(fake_device[:,:,:,:,0], self.minigan_args.output_interimages_dir + \
                    "/fake_samples_epoch%d.png" % current_epoch, normalize=True)



        return disc_err.item(), gen_err.item()

    # RUN
    def run(self):
        if (hvd.rank() == 0):
            print('Hello Run\n')

        run_d_loss = []
        run_g_loss = []

        if (hvd.rank() == 0):
            # Make output directories if they do not exist
            if not os.path.exists(self.minigan_args.output_dir):
                os.makedirs(self.minigan_args.output_dir)

            if not os.path.exists(self.minigan_args.output_interimages_dir):
                os.makedirs(self.minigan_args.output_interimages_dir)

        hvd.allreduce(torch.tensor(0), name="barrier")

        if (hvd.rank() == 0):
            print('Epochs: ' + str(self.minigan_args.epochs))

        for epoch in range(self.minigan_args.epochs):

            d_loss, g_loss = self.run_epoch(current_epoch = epoch,
                                            num_batches   = self.minigan_args.num_images // self.minigan_args.batch_size)

            run_d_loss.append(d_loss)
            run_g_loss.append(g_loss)

        np.save(self.minigan_args.output_dir + '/d_loss_' + str(self.minigan_args.dim_mode) + 'd', run_d_loss)
        np.save(self.minigan_args.output_dir + '/g_loss_' + str(self.minigan_args.dim_mode) + 'd', run_g_loss)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Will throw warnings for torch <= 1.3.1
            torch.save(self.discriminator, self.minigan_args.output_dir + '/discriminator_model')
            torch.save(self.generator, self.minigan_args.output_dir + '/generator_model')
