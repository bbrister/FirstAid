"""
Test the data augmentation in augment3d.py

Usage: python test_augment3d.py [input.nii.gz]
"""

import sys
import numpy as np
import nibabel as nib

import augment3d

outName = 'out.nii.gz'

inPath = sys.argv[1]

# Load the image
im = nib.load(inPath)
data = im.get_data()

# Test the augmenter with each transform
identity = augment3d.cuda_affine_augment3d(data)
nib.save(nib.Nifti1Image(identity, im.affine), 'identity.nii.gz')

rotate = augment3d.cuda_affine_augment3d(data, rotMax=(90, 90, 90)) 
nib.save(nib.Nifti1Image(rotate, im.affine), 'rotate.nii.gz')

reflect = augment3d.cuda_affine_augment3d(data, pReflect=(0, 0, 1))
nib.save(nib.Nifti1Image(reflect, im.affine), 'reflect.nii.gz')

shear = augment3d.cuda_affine_augment3d(data, shearMax=(2,0,0))
nib.save(nib.Nifti1Image(shear, im.affine), 'shear.nii.gz')

translate = augment3d.cuda_affine_augment3d(data, transMax=(30,30,30)) 
nib.save(nib.Nifti1Image(translate, im.affine), 'translate.nii.gz')

other = augment3d.cuda_affine_augment3d(data, otherScale=0.33)
nib.save(nib.Nifti1Image(other, im.affine), 'other.nii.gz')
