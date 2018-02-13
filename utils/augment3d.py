import math
import numpy as np

"""
    Adjust the translation component of an affine transform so that it fixes
    the center of an image with shape 'shape'. Preserves the linear part.
"""
def fix_center_affine(mat, shape):
    mat = mat.astype(float)
    center = (np.array(shape).astype(float) - 1) / 2
    mat[0:3, 3] = center - mat[0:3, 0:3].dot(center[np.newaxis].T).T
    return mat

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
def cuda_affine_augment3d(im, seg=None, rand_seed=None, 
    rotMax=(0, 0, 0), pReflect=(0, 0, 0), shearMax=(1,1,1), transMax=(0,0,0), 
    otherScale=0):

    # Import the required package
    from pyCudaImageWarp import cudaImageWarp

    #  Set the random seed, if specified
    if rand_seed is not None:
        np.random.seed(rand_seed)

    # ---Randomly generate the desired transforms, in homogeneous coordinates---

    # Uniform rotation
    rotate_deg = np.random.uniform(low=-np.array(rotMax), high=rotMax)
    lin_rotate = np.identity(3)
    for i in range(3): # Rotate about each axis and combine
        # Compute the angle of rotation, in radians
        deg = np.random.uniform(low=-rotMax[i], high=rotMax[i])
        rad = deg * 2 * math.pi / 360

        # Form the rotation matrix about this axis
        rot = np.identity(3)
        axes = [x for x in range(3) if x != i]
        rot[axes[0], axes[0]] = math.cos(rad)
        rot[axes[0], axes[1]] = -math.sin(rad)
        rot[axes[1], axes[0]] = -rot[axes[0], axes[1]]
        rot[axes[1], axes[1]] = rot[axes[0], axes[0]]

        # Compose all the rotations
        lin_rotate = lin_rotate.dot(rot)

    # Extend the linear rotation to an affine transform
    mat_rotate = np.identity(4)
    mat_rotate[0:3, 0:3] = lin_rotate
    mat_rotate = fix_center_affine(mat_rotate, im.shape)

    # Uniform shear, same chance of shrinking and growing
    if np.any(shearMax <= 0):
        raise ValueError("Invalid shearMax: %f" % (shear))    
    shear = np.random.uniform(low=1.0, high=shearMax, size=3)
    invert_shear = np.random.uniform(size=3) < 0.5
    shear[invert_shear] = 1.0 / shear[invert_shear]
    mat_shear = fix_center_affine(np.diag(np.hstack((shear, 1))), im.shape)

    # Reflection
    do_reflect = np.random.uniform(size=3) < pReflect
    mat_reflect = fix_center_affine(np.diag(np.hstack((1 - 2 * do_reflect, 1))),
        im.shape)

    # Generic affine transform, Gaussian-distributed
    mat_other = np.identity(4)
    mat_other[0:3, :] = mat_other[0:3, :] + \
        np.random.normal(loc=0.0, scale=otherScale, size=(3,4))
    mat_other = fix_center_affine(mat_other, im.shape) 

    # Uniform translation
    mat_translate = np.identity(4)
    mat_translate[0:3, 3] = np.random.uniform(low=-np.array(transMax), 
        high=transMax)

    # Compose all the transforms
    warp_affine = (
        mat_translate.dot( mat_rotate.dot( mat_shear.dot( mat_reflect.dot(
                mat_other))))
    )[0:3, :]

    # Warp the image
    im_warp = cudaImageWarp.cudaImageWarp(im, warp_affine, interp='linear')

    # Return early if there's no segmentation
    if seg is None:
        return im_warp

    # Warp the segmentation
    seg_warp = cudaImageWarp.cudaImageWarp(seg, warp_affine, interp='nearest')
    return im_warp, seg_warp
