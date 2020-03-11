# miniGAN

miniGAN is a generative adversarial network code developed as part of the 
Exascale Computing Project's (ECP) [ExaLearn](https://www.exascaleproject.org/project/exalearn-co-design-center-for-exascale-machine-learning-technologies/) 
project at Sandia National Laboratories.

miniGAN v. 1.0.0

For questions, contact J. Austin Ellis (johelli@sandia.gov) or Siva Rajamanickam (srajama@sandia.gov).

------------------------------------------------
Description:
------------------------------------------------

miniGAN is a proxy application for generative adversarial networks.

The objective of the miniGAN miniapp is to model performance for training 
generator and discriminator networks.
The GAN's generator and discriminator generate plausible 2D/3D maps and identify fake maps, respectively. 
Related applications exist in cosmology (CosmoFlow, ExaGAN) and wind energy (ExaWind).

Authors: J. Austin Ellis (johelli@sandia.gov) and Siva Rajamanickam (srajama@sandia.gov)

------------------------------------------------
Benchmarks:
------------------------------------------------
Coming soon.

------------------------------------------------
To Install:
------------------------------------------------

### Model/Package Combos

#### Pytorch Tested
- python/3.5.2
- pytorch/1.3.1
- horovod/0.18.2

### Install

1. Enter desired pytorch directory and prepare python env
  + Run `./setup_python_env.sh`
  + Run `source ./minigan_torch_env/bin/activate` (for pytorch)
  + (Run `deactivate` to exit python env)

2. Generate bird dataset
  + Move to ${minigan_dir}/data 
  + Run `python generate_random_images.py --help` for running options

3. Run `python3 minigan_driver.py --help` for running options 
  + Default dataset is "random". Switch to "--dataset bird" to use generated dataset.

4. Run!

### OLCF Summit instructions

1. Do not run `setup_python_env.sh`, instead run `module load ibm-wml-ce/1.6.2-0` to load IBM's Watson ML Community Edition.
    + This should contain PyTorch, Horovod, Matplotlib 
    + Have not been successful with Summit's standalone pip / anaconda
2. Obtain an interactive session using
    + `bsub --nnodes 1 -Is -W 1:00 -P <ProjID> /bin/bash`
3. Run using 
    + `ddlrun python3 minigan_driver.py --dataset bird`


### Experimental! 
#### Make use of kokkos-kernels layers/operations

1. In development

Please report bugs or feature requests to: https://github.com/SandiaMLMiniApps/miniGAN/issues

##### [LICENSE](https://github.com/SandiaMLMiniApps/miniGAN/blob/devel/LICENSE)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

miniGAN is licensed under standard 3-clause BSD terms of use.  For
specifics, please refer to the LICENSE file contained in the
repository or distribution.  Under the terms of Contract DE-NA0003525 with NTESS,
the U.S. Government retains certain rights in this software.













