##########################################################################
# ************************************************************************
#
#               miniGAN : GAN proxy application
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

# general imports
import sys, os
import timeit
import argparse

# data and ml imports
import numpy as np

# Torch
import torch
import horovod.torch as hvd

sys.path.append('./src')

import build_gan


###############################################################################
                              # MINIGAN MAIN #
###############################################################################

def main():

    main_tic = timeit.default_timer()

    ### Command Line Inputs ###
    parser = argparse.ArgumentParser(description='MiniGAN proxy app')

    # GENERAL OPTIONS
    parser.add_argument('--dataset', type=str, default="random", metavar='Str',
                        help='which dataset to use from ("random" or "bird") (default: "random")')
    parser.add_argument('--dim-mode', type=int, default=2, metavar='N',
                        help='2d or 3d mode (default: 2)')
    parser.add_argument('--kk-mode', action='store_true', default=False,
                        help='use kk convolutions')
#    parser.add_argument('--input-file', type=str, default="none", metavar='Str',
#                        help='which input file in ./data/ to use (default: "none")')

    # TRAINING OPTIONS
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--disc-lr', type=float, default=0.0002, metavar='LR',
                        help='discriminator learning rate (default: 0.0004)')
    parser.add_argument('--gen-lr', type=float, default=0.0002, metavar='LR',
                        help='generator learning rate (default: 0.0001)')
    parser.add_argument('--disc-beta1', type=float, default=0.5, metavar='B1',
                        help='discriminator Adam beta parameter (default: 0.5)')
    parser.add_argument('--gen-beta1', type=float, default=0.5, metavar='B1',
                        help='generator Adam beta parameter (default: 0.5)')
    parser.add_argument('--bn-eps', type=float, default=1e-5, metavar='EPS',
                        help='batch norm epsilon (default: 1e-5)')
    parser.add_argument('--bn-mom', type=float, default=.1, metavar='MOM',
                        help='batch norm momentum (default: .1)')
    parser.add_argument('--leaky-relu', type=float, default=.2, metavar='LR',
                        help='negative slope of leaky relu (default: .2)')
#    parser.add_argument('--disc-clamp', type=float, default=0.01, metavar='DC',
#                        help='clamp discriminator weights during training (default: 0.01)')
    parser.add_argument('--soft-label', type=float, default=0.0, metavar='LS',
                        help='soften labels from 0/1 to LS/(1-LS) (default: 0.0)')
    parser.add_argument('--label-mut', type=float, default=0.0, metavar='LM',
                        help='discriminator label mutation rate (default: 0.0)')
    parser.add_argument('--dg-train-ratio', type=int, default=1, metavar='TR',
                        help='train discriminator N times for each generator \
                              training (default: 1)')


    # NETWORK INPUT OPTIONS
    parser.add_argument('--num-images', type=int, default=64, metavar='N',
                        help='number of images in training set (default: 64)')
    parser.add_argument('--image-dim', type=int, default=64, metavar='N',
                        help='dimensions of images (square/cube) (default: 64)')
    parser.add_argument('--num-channels', type=int, default=1, metavar='N',
                        help='number of input channels (default: 1)')
    parser.add_argument('--data-workers', type=int, default=2, metavar='N',
                        help='number of data loader workers (default: 2)')


    # DISCRIMINATOR OPTIONS
    parser.add_argument('--disc-layers', type=int, default=3, metavar='N',
                        help='number of discriminator convolutions (default: 3)')
    parser.add_argument('--disc-kern-size', type=int, default=4, metavar='N',
                        help='size of discriminator filter kernel (default: 4)')
    parser.add_argument('--disc-filters', type=int, default=64, metavar='N',
                        help='number of discriminator filters (default: 64)')
    parser.add_argument('--disc-kern-stride', type=int, default=2, metavar='N',
                        help='stride of discriminator filter kernel (default: 2)')
    parser.add_argument('--disc-padding', type=int, default=1, metavar='N',
                        help='image padding for discriminator convolutions (default: 1)')


    # GENERATOR OPTIONS
    parser.add_argument('--gen-layers', type=int, default=3, metavar='N',
                        help='number of generator transp convolutions (default: 3)')
    parser.add_argument('--gen-kern-size', type=int, default=4, metavar='N',
                        help='size of generator filter kernel (default: 4)')
    parser.add_argument('--gen-filters', type=int, default=64, metavar='N',
                        help='number of generator filters (default: 64)')
    parser.add_argument('--gen-kern-stride', type=int, default=2, metavar='N',
                        help='stride of generator filter kernel (default: 2)')
    parser.add_argument('--gen-padding', type=int, default=1, metavar='N',
                        help='image passing for generator transposed convolutions (default: 1)')
    parser.add_argument('--gen-noise', type=int, default=100, metavar='N',
                        help='size of generator noise vector input (default: 100)')

    # DIRECTORY OPTIONS
    parser.add_argument('--output-dir', type=str, default="./output", metavar='Str',
                        help='output directory (default: "./output")')
    parser.add_argument('--data-dir', type=str, default="../data/bird_images", metavar='Str',
                        help='data directory (default: "../data/bird_images")')


    # CHECKPOINT/LOGGING OPTIONS
    parser.add_argument('--checkpoint', type=int, default=1, metavar='N',
                        help='epochs between checkpoints (default: 1)')
    parser.add_argument('--log-interval', type=int, default=32, metavar='N',
                        help='batches between logging training status (default: 32')
    parser.add_argument('--checkpoint-images', type=int, default=16, metavar='N',
                        help='number of images to save during checkpoint (default: 16')

    # PROFILER OPTIONS
    parser.add_argument('--profile', action='store_true', default=False,
                        help='profile layers instead of running GAN training')
    parser.add_argument('--prof-steps', type=int, default=3, metavar='N',
                        help='number of dummy steps in profiler (default: 3)')
    parser.add_argument('--prof-images', type=int, default=1, metavar='N',
                        help='number of images to profile (default: 1)')


    # ARCH ENV OPTIONS
    parser.add_argument('--num-threads', type=int, default=32, metavar='N',
                        help='number of threads to use for CPU (default: 32)')
    parser.add_argument('--num-gpus', type=int, default=1, metavar='N',
                        help='number of gpus available (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # HOROVOD OPTIONS
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')

    # OTHER OPTIONS
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    # Parse User Arguments
    minigan_args = parser.parse_args()


    minigan_args.cuda = not minigan_args.no_cuda and torch.cuda.is_available()


    # Input File (if using)
#    if (minigan_args.input_file != "none")
#        minigan_args = minigan_read_input(minigan_args)

    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(minigan_args.seed)

    if (hvd.rank() == 0):
        print('\n-----------------------------------------\n')
        print('**** BEGIN miniGAN PROXY APPLICATION ****')
        print('\n-----------------------------------------\n')
        print('\nRunning with %d ranks\n\n' % hvd.size())

    if (minigan_args.batch_size < hvd.size()):
        print("Changing batch_size from %d to %d (number of ranks)" % (minigan_args.batch_size, hvd.size()))
        minigan_args.batch_size = hvd.size()

    minigan_args.test_batch_size = minigan_args.batch_size

    if minigan_args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(minigan_args.seed)
        minigan_args.device = "cuda"
    else:
        minigan_args.device = "cpu"

    # Print list of command line arguments
    if (hvd.rank() == 0):
        print("Parser Arguments")
        for arg in vars(minigan_args):
            print("%s: %s" % (arg, getattr(minigan_args, arg)))

    if(hvd.rank() == 0 and not minigan_args.cuda):
        print("Running miniGAN with %d threads" % (minigan_args.num_threads))

    torch.set_num_threads(minigan_args.num_threads)

###############################################################################

    os.environ["OMP_NUM_THREADS"] = str(minigan_args.num_threads)
    os.environ["OMP_PROC_BIND"]   = "spread"
    os.environ["OMP_PLACES"]      = "threads"

    os.environ["KMP_BLOCKTIME"]  = "30"
    os.environ["KMP_SETTINGS"]   = "1"
    os.environ["KMP_AFFINITY"]   = "granularity=fine,verbose,compact,1,0"

###############################################################################

    if (hvd.rank() == 0):
        print('\n---BEGIN SETUP---\n')

    setup_tic = timeit.default_timer()

    # build GAN with miniGAN user args
    GAN = build_gan.miniGAN(minigan_args)

    setup_toc = timeit.default_timer()

    if (hvd.rank() == 0):
        print('\n---SETUP DONE---\n')

###############################################################################

    if (hvd.rank() == 0):
        print('\n---BEGIN TRAINING---\n')

    run_tic = timeit.default_timer()

    ###############
    ### GAN RUN ###
    ###############

    GAN.run()

    ###############

    run_toc = timeit.default_timer()

    if (hvd.rank() == 0):
        print('\n---TRAINING DONE---\n')



###############################################################################

    ### FINISH ###

    if (hvd.rank() == 0):
        print('-----DONE-----')

    main_toc = timeit.default_timer()

    if (hvd.rank() == 0):
        print("\n\n-------------------\n",
            "miniGAN Setup Time: ",
            setup_toc - setup_tic,
            "\n-------------------\n")

        print("\n-------------------\n",
            "miniGAN Training Time: ",
            run_toc - run_tic,
            "\n-------------------\n")

        print("\n-------------------\n",
            "miniGAN Total Runtime: ",
            main_toc - main_tic,
            "\n-------------------\n")

## RUN ###
if __name__ == "__main__":
    main()
