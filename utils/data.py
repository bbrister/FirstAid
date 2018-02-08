import csv
import h5py
import matplotlib.animation as animation
import numpy as np
from os import listdir, remove, mkdir
from os.path import isfile, join, isdir
from pylab import *
import scipy.misc
import tensorflow as tf
import socket
import sys
import time

def find_data_shape(path_data):
    """
    Reads in one piece of data to find out number of channels.
    INPUT:
    path_data - (string) path of data
    """
    statement = ''
    dir_data = listdir(path_data)
    matrix_size = 0
    num_channels = 0
    # Trying to look into each patient folder.
    for folder_data in dir_data:
        path_patient = join(path_data, folder_data)
        if not isdir(path_patient):
            continue
        dir_file = listdir(path_patient)
        # Trying to look at each image file.
        for name_file in dir_file:
            if name_file[-3:] != '.h5':
                continue
            path_file = join(path_patient, name_file)
            try:
                with h5py.File(path_file) as hf:
                    img = np.array(hf.get('data'))
                    matrix_size = img.shape[0]
                    num_channels = img.shape[-1]
            except:
                statement += path_file + ' is not valid.\n'
            if matrix_size != 0:
                break
        if matrix_size != 0:
            break
    if matrix_size == 0:
        statement += "Something went wrong in finding out img dimensions.\n"
    sys.stdout.write(statement)
    return matrix_size, num_channels

def calculate_iters(data_count, epoch, batch_size):
    """
    Uses training path, max_epoch, and batch_size to calculate
    the number of iterations to run, how long an epoch is in
    iterations, and how often to print.
    INPUT:
    data_count - (int) length of data
    epoch - (int) max number of epochs
    batch_size - (int) size of batch
    """
    iter_count = int(np.ceil(float(epoch) * data_count / batch_size))
    epoch_every = int(np.ceil(float(iter_count) / epoch))
    print_every = min([10000, epoch_every])
    print_every = max([100, print_every])
    return iter_count, epoch_every, print_every

def data_augment(data_iter, data_seg=None, rand_seed=None):
    """
    Stochastically augments the single piece of data.
    INPUT:
    - data_iter: (3d ND-array) the single piece of data
    - data_seg: (2d ND-array) the corresponding segmentation
    """
    matrix_size = data_iter.shape[0]
    # Setting Seed
    if rand_seed is not None:
        np.random.seed(rand_seed)
    # Creating Random variables
    roller = np.round(float(matrix_size/7))
    ox, oy = np.random.randint(-roller, roller+1, 2)
    do_flip = np.random.randn() > 0
    num_rot = np.random.choice(4)
    pow_rand = np.clip(0.05*np.random.randn(), -.2, .2) + 1.0
    add_rand = np.clip(np.random.randn() * 0.1, -.4, .4)
    # Rolling
    data_iter = np.roll(np.roll(data_iter, ox, 0), oy, 1)
    if np.any(data_seg):
        data_seg = np.roll(np.roll(data_seg, ox, 0), oy, 1)
    # Left-right Flipping
    if do_flip:
        data_iter = np.fliplr(data_iter)
        if np.any(data_seg):
            data_seg = np.fliplr(data_seg)
    # Random 90 Degree Rotation
    data_iter = np.rot90(data_iter, num_rot)
    if np.any(data_seg):
        data_seg = np.rot90(data_seg, num_rot)
    # Raising/Lowering to a power
    #data_iter = data_iter ** pow_rand
    # Random adding of shade.
    data_iter += add_rand
    if np.any(data_seg):
        return data_iter, data_seg
    return data_iter

def data_augment3d(data_iter, data_seg=None, rand_seed=None):
    """
    Stochastically augments the single piece of data.
    INPUT:
    - data_iter: (3d ND-array) the single piece of data
    - data_seg: (2d ND-array) the corresponding segmentation
    """
    matrix_size = data_iter.shape[0]
    # Setting Seed
    if rand_seed is not None:
        np.random.seed(rand_seed)
    # Creating Random variables
    roller = np.round(float(matrix_size/7))
    ox, oy, oz = np.random.randint(-roller, roller+1, 3)
    do_flip = np.random.randn() > 0
    num_rot = np.random.choice(4)
    pow_rand = np.clip(0.05*np.random.randn(), -.2, .2) + 1.0
    add_rand = np.clip(np.random.randn() * 0.1, -.4, .4)
    # Rolling
    data_iter = np.roll(np.roll(np.roll(data_iter, ox, 0), oy, 1), oz, 2)
    if np.any(data_seg):
        data_seg = np.roll(np.roll(np.roll(data_seg, ox, 0), oy, 1), oz, 2)
    # Left-right Flipping
    if do_flip:
        data_iter = np.fliplr(data_iter)
        if np.any(data_seg):
            data_seg = np.fliplr(data_seg)
    # Random 90 Degree Rotation
    data_iter = np.rot90(data_iter, num_rot)
    if np.any(data_seg):
        data_seg = np.rot90(data_seg, num_rot)
    # Raising/Lowering to a power
    #data_iter = data_iter ** pow_rand
    # Random adding of shade.
    data_iter += add_rand
    if data_seg is not None:
        return data_iter, data_seg
    return data_iter

def cuda_affine_augment3d(data_iter, data_seg=None, rand_seed=None, 
    rotMax=(0, 0, 0), pReflect=(0, 0, 0), shearMax=(0,0,0), transMax=(0,0,0), 
    otherScale=(0,0,0)):

    # Import the required package
    from pyCudaImageWarp import cudaImageWarp

    #  Set the random seed, if specified
    if rand_seed is not None:
        np.random.seed(rand_seed)

    # ---Randomly generate the desired transforms, in homogeneous coordinates---

    # Uniform rotation
    rotate_deg = np.random.uniform(low=-rotMax, hi=rotMax, size(3, 1))
    lin_rotate = np.identity(3)
    for i in range(3): # Rotate in each dimension and combine
        # Compute the amount of rotation, in radians
        deg = np.random.uniform(low=-rotMax[i], hi=rotMax[i])
        rad = deg * 2 * math.pi / 360

        # Form the rotation matrix about this axis
        rot = np.identity(3)
        axes = [1, 2, 3].remove(i)
        rot[axes[1], axes[1]] = cos(rotRad)
        rot[axes[1], axes[2]] = -sin(rotRad)
        rot[axes[2], axes[1]] = -rot[axes[1], axes[2]]
        rot[axes[2], axes[2]] = rot[axes[1], axes[1]]

        # Compose all the rotations
        lin_rotate = lin_rotate * rot

    # Extend the linear rotation to an affine transform
    mat_rotate = np.identity(4)
    mat_rotate[1:3, 1:3] = lin_rotate

    # Adjust for translation of the center, so we're rotating about the center
    center_coord = float(data_iter.shape - 1) / 2;
    mat_rotate[1:3, 4] = -lin_rotate * center_coord;

    # Uniform shear
    shear = np.random.uniform(low=-max(shearMax, 0.9) hi=shearMax, size(3, 1))
    mat_shear = np.diag(np.hstack(1 + shear, 1))

    # Uniform translation
    mat_translate = np.identity(4)
    mat_translate[1:3, 4] = np.random.uniform(low=-transMax, hi=transMax, size(3, 1))

    # Reflection
    do_reflect = np.random.uniform(size=(1,3)) < pReflect
    mat_reflect = np.diag(np.hstack(1 - 2 * do_reflect, 1))

    # Generic affine transform, Gaussian-distributed
    affSize=(4,4)
    mat_other = np.diag(np.random.normal(loc=1.0,scale=otherScale, affSize))

    # Compose all the transforms
    mat_all = mat_rotate * mat_shear * mat_translate * mat_reflect * mat_other

    # Warp the image
    data_iter_warp = cudaImageWarp.cudaImageWarp(data_iter, mat_all, 
        interp='linear')

    # Return early if there's no segmentation
    if data_seg is None:
        return data_iter_warp

    # Warp the segmentation
    data_seg_warp = cudaImageWarp.cudaImageWarp(data_seg, mat_all, 
        interp='nearest')
    return data_iter_warp, data_seg_warp
