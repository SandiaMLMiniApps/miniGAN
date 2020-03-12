from __future__ import print_function, division

import numpy as np
import os
import argparse
import matplotlib.pylab as plt

# Num pixels in X/Y dim for bird stamp
bird_dim_x = 23
bird_dim_y = 20
# Magnitude of stamp
bird_weight = 10.0

parser = argparse.ArgumentParser(description="Generate Sandia Bird Dataset")

parser.add_argument('--dim-mode', type=int, default=2, metavar='N',
                                help='2d or 3d mode (default: 2)')
parser.add_argument('--num-images', type=int, default=64, metavar='N',
                                help='number of images to create (default: 64)')
parser.add_argument('--image-dim', type=int, default=64, metavar='N',
                                help='dimensions of images (square/cube) (default: 64)')
parser.add_argument('--num-channels', type=int, default=1, metavar='N',
                                help='number of input channels (default: 1)')
parser.add_argument('--data-type', type=str, default="float32", metavar='Str',
                help='data type precision (default: "float32")')
parser.add_argument('--channels-last', action='store_true', default=False,
                help='channels as last dimension (default: channels first)')


parser.add_argument('--output-dir', type=str, default="./bird_images", metavar='Str',
                help='output direectory (default: "bird_images")')

args = parser.parse_args()

size_x = args.image_dim
size_y = args.image_dim
size_z = args.image_dim

if (args.dim_mode == 2):
    if (args.channels_last):
        filename = args.output_dir + '/minigan_bird_' + \
                                str(args.num_images) + 'imgs_' + \
                                str(size_x) + 'x' + \
                                str(size_y) + 'px_' + \
                                str(args.num_channels) + 'chnls'
    # Pytorch Default
    else:
        filename = args.output_dir + '/minigan_bird_' + \
                                str(args.num_images) + 'imgs_' + \
                                str(args.num_channels) + 'chnls_' + \
                                str(size_x) + 'x' + \
                                str(size_y) + 'px'

elif (args.dim_mode == 3):
    if (args.channels_last):
        filename = args.output_dir + '/minigan_bird_' + \
                                str(args.num_images) + 'imgs_' + \
                                str(size_x) + 'x' + \
                                str(size_y) + 'x' + \
                                str(size_z) + 'px_' + \
                                str(args.num_channels) + 'chnls'
    # Pytorch Default
    else:
        filename = args.output_dir + '/minigan_bird_' + \
                                str(args.num_images) + 'imgs_' + \
                                str(args.num_channels) + 'chnls_' + \
                                str(size_x) + 'x' + \
                                str(size_y) + 'x' + \
                                str(size_z) + 'px'
else:
    print("Error: dim mode must be 2 or 3.")
    exit(0);

if (size_x <= bird_dim_x or size_y <= bird_dim_y):
    print("Error: size of images too small to create.")
    exit(0);



# MAIN
def main():

    print('Creating %dD bird images at %s' % (args.dim_mode, filename))

    rez = int(size_y / bird_dim_y / 5)

    if rez < 1:
        rez = 1

    bird_stamp = create_bird_stamp(rez)

    # Make random noise background for images
    if (args.dim_mode == 2):
        if (args.channels_last):
            images = np.random.rand(args.num_images, size_x, size_y, args.num_channels)
        else:
            images = np.random.rand(args.num_images, args.num_channels, size_x, size_y)

    else:
        if (args.channels_last):
            images = np.random.rand(args.num_images, size_x, size_y, size_z, args.num_channels)
        else:
            images = np.random.rand(args.num_images, args.num_channels, size_x, size_y, size_z)

    if (args.data_type == "float32"):
        images = np.float32(images)
    elif (args.data_type == "float16"):
        print("Float16 is currently not supported. Exiting.")
        exit(0);

    print("Making...")
    for i in range(args.num_images):
        if (i % 10 == 0):
            print('Image ' + str(i))

        # Randomly position the Bird Stamp
        x_pos = np.random.randint(0,size_x - bird_stamp.shape[0])
        y_pos = np.random.randint(0,size_y - bird_stamp.shape[1])

        x_fin = x_pos + bird_stamp.shape[0]
        y_fin = y_pos + bird_stamp.shape[1]

        for c in range(args.num_channels):
            if (args.dim_mode == 2):
                if (args.channels_last):
                    images[i, x_pos:x_fin, y_pos:y_fin, c] += bird_weight * bird_stamp[:,:]
                else:
                    images[i, c, x_pos:x_fin, y_pos:y_fin] += bird_weight * bird_stamp[:,:]
            else:
                for z in range(size_z):
                    if (args.channels_last):
                        images[i, x_pos:x_fin, y_pos:y_fin, z, c] += bird_weight * bird_stamp[:,:]
                    else:
                        images[i, c, x_pos:x_fin, y_pos:y_fin, z] += bird_weight * bird_stamp[:,:]



    images /= bird_weight + 1

#    plt.figure()
    f, axarr = plt.subplots(2,2)

    if (args.dim_mode == 2):
        if (args.channels_last):
            axarr[0,0].imshow(images[0,:,:,0])
            axarr[0,1].imshow(images[1,:,:,0])
            axarr[1,0].imshow(images[2,:,:,0])
            axarr[1,1].imshow(images[3,:,:,0])
        else:
            axarr[0,0].imshow(images[0,0,:,:])
            axarr[0,1].imshow(images[1,0,:,:])
            axarr[1,0].imshow(images[2,0,:,:])
            axarr[1,1].imshow(images[3,0,:,:])

    else:
        if (args.channels_last):
            axarr[0,0].imshow(images[0,:,:,0,0])
            axarr[0,1].imshow(images[1,:,:,0,0])
            axarr[1,0].imshow(images[2,:,:,0,0])
            axarr[1,1].imshow(images[3,:,:,0,0])
        else:
            axarr[0,0].imshow(images[0,0,:,:,0])
            axarr[0,1].imshow(images[1,0,:,:,0])
            axarr[1,0].imshow(images[2,0,:,:,0])
            axarr[1,1].imshow(images[3,0,:,:,0])



#    plt.show()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    f.savefig(args.output_dir + '/bird_images_%dd.png' % (args.dim_mode))
    np.save(filename, images, allow_pickle=True)

    print('Success!')



# Create Sandia Bird Stamp
# https://www.sandia.gov/about/history/_assets/images/timeline/snl_logo_70-present.png
def create_bird_stamp(rez):

    print('\n\nCreating Bird Stamp\n')

    stamp_dim_x = rez * bird_dim_x
    stamp_dim_y = rez * bird_dim_y

    bird_stamp = \
        np.ones([stamp_dim_x, stamp_dim_y], dtype=args.data_type);

    # Orientation: (0,0) is in the upper left corner

    # Upper Left Corner
    bird_stamp[0:2*rez,   0:rez] = 0
    bird_stamp[0:rez,   rez:2*rez] = 0

    # Lower Right Corner
    bird_stamp[-2*rez:,    -rez:] = 0
    bird_stamp[   -rez:, -2*rez:-rez] = 0

    # Upper Right Corner
    bird_stamp[0:2*rez,   -rez:] = 0
    bird_stamp[0:rez,   -2*rez:-rez] = 0

    # Lower Left Corner
    bird_stamp[-2*rez:,    0:rez] = 0
    bird_stamp[   -rez:, rez:2*rez] = 0


    # Left/Right Strip
    bird_stamp[3*rez:-3*rez,  2*rez:3*rez] = 0
    bird_stamp[3*rez:-3*rez, -3*rez:-2*rez] = 0

    # Top/Bottom Strip
    bird_stamp[ 2*rez:3*rez,  3*rez:-3*rez] = 0
    bird_stamp[-3*rez:-2*rez, 3*rez:-3*rez] = 0

    # Left Wing
    bird_stamp[ -6*rez:-3*rez, 3*rez:7*rez] = 0
    bird_stamp[-10*rez:-6*rez, 5*rez:7*rez] = 0
    bird_stamp[-12*rez:-8*rez, 6*rez:8*rez] = 0

    # Right Wing
    bird_stamp[ -6*rez:-3*rez, -7*rez:-3*rez] = 0
    bird_stamp[-10*rez:-6*rez, -7*rez:-5*rez] = 0
    bird_stamp[-12*rez:-8*rez, -8*rez:-6*rez] = 0

    # Left of Head
    bird_stamp[3*rez:5*rez, 3*rez:6*rez] = 0
    bird_stamp[5*rez:7*rez, 3*rez:8*rez] = 0

    # Right of Head
    bird_stamp[3*rez:7*rez, -8*rez:-3*rez] = 0

    for i in range(stamp_dim_x):
        if (i % rez != 0):
            continue;
        print("\nRow %5d:    " % i, end=" ")
        for j in range(stamp_dim_y):
            if (j % rez != 0):
                continue;

            print("%3d" % (bird_weight * bird_stamp[i,j]), end=" ")

    print("\n\nBird Stamp Complete!\n\n")

#    print(bird_stamp.shape)

#    plt.imshow(bird_stamp)
#    plt.show()

    return bird_stamp


if __name__ == "__main__":
    main()
