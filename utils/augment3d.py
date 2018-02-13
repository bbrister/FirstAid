import numpy as np

"""
    Adjust the translation component of an affine transform so that it fixes
    the center of an image with shape 'shape'. Preserves the linear part.
"""
def fix_center_affine(mat, shape):
    mat = mat.astype(float)
    center = float(shape - 1) / 2
    mat[1:3, 4] = -mat[1:3, 1:3] * center

"""
    Randomly generates a 3D affine map based on the parameters given. Then 
    applies the map to warp the input image and, optionally, the segmentation.
    Warping is done on the GPU using pyCudaImageWarp.

    By default, the function only generates the identity map. The affine
    transform distribution is controlled by the following parameters:
        rotMax - Uniform rotation about (x,y,z) axes. For example, (10,10,10)
            means +-10 degrees in about each axis.
        pReflect - Chance of reflecting about (x,y,z) axis. For example, 
                (.5, 0, 0) means there is a 50% chance of reflecting about the
                x-axis.
        shearMax - Uniform shearing about each axis. For example, (1.1, 1.1, 
                1.1) shears in each axis in the range (1.1, 1 / 1.1)
        transMax - Uniform translation in each coordinate. For example, (10, 10,
                10) translates by at most +-10 voxels in each coordinate.
        otherScale - Gaussian-distributed affine transform. This controls the
                variance of each parameter.

    All transforms fix the center of the image, except for translation.
"""
def cuda_affine_augment3d(data_iter, data_seg=None, rand_seed=None, 
    rotMax=(0, 0, 0), pReflect=(0, 0, 0), shearMax=(0,0,0), transMax=(0,0,0), 
    otherScale=0):

    # Import the required package
    from pyCudaImageWarp import cudaImageWarp

    #  Set the random seed, if specified
    if rand_seed is not None:
        np.random.seed(rand_seed)

    # ---Randomly generate the desired transforms, in homogeneous coordinates---

    # Uniform rotation
    rotate_deg = np.random.uniform(low=-rotMax, hi=rotMax, size=(3, 1))
    lin_rotate = np.identity(3)
    for i in range(3): # Rotate about each axis and combine
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
    mat_rotate = fix_center_affine(mat_rotate)

    # Uniform shear
    shear = np.random.uniform(low=-max(shearMax, 0.9), hi=shearMax, size=(3, 1))
    mat_shear = fix_center_affine(np.diag(np.hstack(1 + shear, 1)))

    # Reflection
    do_reflect = np.random.uniform(size=(1,3)) < pReflect
    mat_reflect = fix_center_affine(np.diag(np.hstack(1 - 2 * do_reflect, 1)))

    # Generic affine transform, Gaussian-distributed
    mat_other = np.identity(4)
    mat_other[1:3, :] = mat_other[1:3, :] + \
        np.random.normal(loc=0.0,scale=otherScale, size=(3,4))
    mat_other = fix_center_affine(mat_other) 

    # Uniform translation
    mat_translate = np.identity(4)
    mat_translate[1:3, 4] = np.random.uniform(low=-transMax, hi=transMax, 
        size=(3, 1))

    # Compose all the transforms
    warp_affine = (
        mat_translate * mat_rotate * mat_shear * mat_reflect * mat_other
    )[1:3, :]

    # Warp the image
    data_iter_warp = cudaImageWarp.cudaImageWarp(data_iter, warp_affine, 
        interp='linear')

    # Return early if there's no segmentation
    if data_seg is None:
        return data_iter_warp

    # Warp the segmentation
    data_seg_warp = cudaImageWarp.cudaImageWarp(data_seg, warp_affine,
        interp='nearest')
    return data_iter_warp, data_seg_warp
