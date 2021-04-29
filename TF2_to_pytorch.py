import argparse
import os
import sys
import time
import matplotlib.pyplot as plt


import numpy as np
import tensorflow as tf
import torch

# Booleans d'étape
PYTORCH = True
TF2 = True
recompute_x = False
compare = True

# Group
group = "S4"

# Inputs
kernel_size = 3
n_in = 1
n_out = 1
ds = 0
is_training = True
use_bn = False # Turns batch normalization on or off


if PYTORCH:
    torch.manual_seed(1)
if TF2:
    tf.random.set_seed(1)

# Compute x or load it
if recompute_x:
    shape = (16, 16, 16)
    x = tf.random.normal(shape)

    print("Initial x values")
    print(x.shape)
    print("-----------\n")

    np.save("./npys/x.npy", x, allow_pickle=False)

    print("Saved x as a file")
else:
    # Load
    x = tf.convert_to_tensor(np.load("./npys/x.npy"))
    x_p = torch.from_numpy(np.load("./npys/x.npy"))

    x2 = tf.convert_to_tensor(np.load("./npys/x.npy"))
    x2_p = torch.from_numpy(np.load("./npys/x.npy"))
    print("\n Adding Tensorflow dimensions")

    # Adding dimensions
    x = tf.expand_dims(x, axis=0)
    x = tf.expand_dims(x, axis=4)

    x_p = x_p.unsqueeze(0)
    x_p = x_p.unsqueeze(0)

    x2 = tf.expand_dims(x2, axis=0)
    x2 = tf.expand_dims(x2, axis=4)

    x2_p = x2_p.unsqueeze(0)
    x2_p = x2_p.unsqueeze(0)
    print(f"Loaded x (torch) of shape {x_p.shape}")
    print(f"Loaded x (tf) of shape {x.shape}")

## TF2
if TF2:
    sys.path.append('./cubenet')
    from cubenet.layers import Layers as Layers_v2

    layers_v2 = Layers_v2(group)
    group_dim_v2 = layers_v2.group_dim

    x_TF2 = layers_v2.conv_block(x, kernel_size, n_out, is_training, use_bn=use_bn)
    x_TF2_2 = layers_v2.conv_block(x_TF2, kernel_size, n_out, is_training, use_bn=use_bn)
    
    x = tf.expand_dims(x, -1)
    x2 = tf.expand_dims(x2, -1)
    
    Gx_TF2 = layers_v2.Gconv_block(x, kernel_size, n_out, is_training, drop_sigma=ds, use_bn=use_bn)
    Gx_TF2_2 = layers_v2.Gconv_block(Gx_TF2, kernel_size, n_out, is_training, drop_sigma=ds, use_bn=use_bn)

    print("\n ------After transformations TF2----------")
    print(f"conv : {x_TF2.shape}")
    print(f"G-conv {Gx_TF2_2.shape}")

    np.save("./npys/convTF2.npy", x_TF2, allow_pickle=False)
    np.save("./npys/convTF2_2.npy", x_TF2_2, allow_pickle=False)
    np.save("./npys/gconvTF2.npy", Gx_TF2, allow_pickle=False)
    np.save("./npys/gconvTF2_2.npy", Gx_TF2_2, allow_pickle=False)
    print("Saved the layers results as files")

    sys.path.remove('./cubenet')

## PYTORCH
if PYTORCH:
    sys.path.append('./cubenet_torch')
    from cubenet_torch.layers import Layers as PLayers

    layers_p = PLayers(group)
    group_dim_p = layers_p.group_dim

    x_P = layers_p.conv_block(x_p, kernel_size, n_out, is_training, use_bn=use_bn)
    x_P_2 = layers_p.conv_block(x_P, kernel_size, n_out, is_training, use_bn=use_bn)

    x_p = x_p.unsqueeze(2)
    x2_p = x2_p.unsqueeze(2)

    Gx_P = layers_p.Gconv_block(x_p, kernel_size, n_out, is_training, drop_sigma=ds, use_bn=use_bn) # drop_sigma=ds, use_bn=use_bn
    Gx_P_2 = layers_p.Gconv_block(Gx_P, kernel_size, n_out, is_training, drop_sigma=ds, use_bn=use_bn)

    print("\n ------After transformations Pytorch----------")
    print(f"conv : {x_P.shape}")
    print(f"G-conv {Gx_P.shape}")

    np.save("./npys/convP.npy", x_P.detach().numpy(), allow_pickle=False)
    np.save("./npys/convP_2.npy", x_P_2.detach().numpy(), allow_pickle=False)
    np.save("./npys/gconvP.npy", Gx_P.detach().numpy(), allow_pickle=False)
    np.save("./npys/gconvP_2.npy", Gx_P_2.detach().numpy(), allow_pickle=False)
    print("Saved the layers results as files")
    sys.path.remove('./cubenet_torch')

# Compare PYTORCH to TF2. This assumes that the files have been computed
if compare:
    # Loading
    convP = tf.convert_to_tensor(np.load("./npys/convP.npy"))
    convTF2 = tf.convert_to_tensor(np.load("./npys/convTF2.npy"))
    gconvP = tf.convert_to_tensor(np.load("./npys/gconvP.npy"))
    gconvTF2 = tf.convert_to_tensor(np.load("./npys/gconvTF2.npy"))

    # Extracting what can be compared
    to_compare_P = convP[0,0,:,:,:]
    to_compare_TF2 = convTF2[0,:,:,:,0]
    to_compare_gP = gconvP[0,0,0,:,:,:]
    to_compare_gTF2 = gconvTF2[0,:,:,:,0,0]

    rtol = 1e-2
    atol = 1e-2

    bool_map = tf.experimental.numpy.isclose(to_compare_P, to_compare_TF2, rtol=rtol, atol=atol)
    bool_conv = tf.reduce_all(bool_map)

    gbool_map = tf.experimental.numpy.isclose(to_compare_gP, to_compare_gTF2, rtol=rtol, atol=atol)
    gbool_conv = tf.reduce_all(gbool_map)

    print("\nEgalité de PYTORCH et TF2:")
    print(f"Pour les G-conv {gbool_conv}")
    print(f"Pour les conv {bool_conv}")

    if True:
        fig=plt.figure(figsize=(8, 8))
        for i in range(2):
            fig.add_subplot((i+1), 4, 1+(i*4))
            plt.imshow(to_compare_P[i,:,:])

            fig.add_subplot((i+1), 4, 2+(i*4))
            plt.imshow(to_compare_gP[i,:,:])

            fig.add_subplot((i+1), 4, 3+(i*4))
            plt.imshow(gbool_map[i,:,:])

            fig.add_subplot((i+1), 4, 4+(i*4))
            plt.imshow(x[0,i,:,:,0])
        plt.show()
    
if False:
    
    ## Pour PYTORCH
    if PYTORCH:
        gconvP_2 = tf.convert_to_tensor(np.load("./npys/gconvP_2.npy"))
        gconvP = tf.convert_to_tensor(np.load("./npys/gconvP.npy"))
        convP_2 = tf.convert_to_tensor(np.load("./npys/convP_2.npy"))
        convP = tf.convert_to_tensor(np.load("./npys/convP.npy"))

        bool_g_conv_P = tf.reduce_all(tf.math.equal(gconvP, gconvP_2))
        bool_conv_P = tf.reduce_all(tf.math.equal(convP, convP_2))

        print("\nEgalité de PYTORCH et un autre PYTORCH:")
        print(f"Pour les G-conv {bool_g_conv_P}")
        print(f"Pour les conv {bool_conv_P}")

    if TF2:
        ## Pour TF2

        gconvTF2 = tf.convert_to_tensor(np.load("./npys/gconvTF2.npy"))
        gconvTF2_2 = tf.convert_to_tensor(np.load("./npys/gconvTF2_2.npy"))
        convTF2 = tf.convert_to_tensor(np.load("./npys/convTF2.npy"))
        convTF2_2 = tf.convert_to_tensor(np.load("./npys/convTF2_2.npy"))

        bool_g_conv = tf.reduce_all(tf.equal(gconvTF2_2, gconvTF2))
        bool_conv = tf.reduce_all(tf.equal(convTF2_2, convTF2))
        print("\nEgalité de TF2 et un autre TF2:")
        print(f"Pour les G-conv {(bool_g_conv)}")
        print(f"Pour les conv {(bool_conv)}")

        # plt.imshow(gconvTF2_2[0,:,:,0,0,0])
        # plt.show()
        # plt.imshow(gconvTF2[0,:,:,0,0,0])
        # plt.show()

        # x_compare = tf.convert_to_tensor(np.load("./npys/x.npy"))

        # bool_x = tf.reduce_all(tf.equal(x2, x))
        # print(bool_x)

        # plt.imshow(x[0,:,:,0,0,0])
        # plt.show()
        # plt.imshow(x_compare[0,:,:,0,0])
        # plt.show()
