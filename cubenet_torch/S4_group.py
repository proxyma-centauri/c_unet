"""Models for the RFNN experiments"""
import argparse
import os
import sys
import time

import numpy as np

import torch
import torchvision.transforms.functional as torchtransforms


class S4_group(object):
    def __init__(self):
        self.group_dim = 24
        self.cayleytable = self.get_cayleytable()


    def get_Grotations(self, x):
        """Rotate the tensor x with all 24 S4 rotations

        Args:
            x: [h,w,d,n_channels]
        Returns:
            list of 24 rotations of x [[h,w,d,n_channels],....]
        """
        xsh = x.shape
        angles = [0.,np.pi/2.,np.pi,3.*np.pi/2.]
        rx = []
        for i in range(4):
            # Z4 rotations about the z axis
            perm = [0,2,1,3]
            y = x.permute(perm)
            y = torchtransforms.rotate(y, angles[i])
            y = y.permute(perm)
            # Rotations in the quotient space (sphere S^2)
            # i) Z4 rotations about y axis
            for j in range(4):
                perm = [0,3,2,1]
                z = y.permute(perm)
                z = torchtransforms.rotate(z, angles[-j])
                z = z.permute(perm)
                
                rx.append(z)
            # ii) 2 rotations to the poles about the x axis
            perm = [0,1,3,2]
            z = y.permute(perm)
            z = torchtransforms.rotate(z, angles[3])
            z = z.permute(perm)
            rx.append(z)

            z = y.permute(perm)
            z = torchtransforms.rotate(z, angles[1])
            z = z.permute(perm)
            rx.append(z)

        return rx


    def G_permutation(self, W):
        """Permute the outputs of the group convolution"""
        Wsh = W.shape
        cayley = self.cayleytable
        U = []
        for i in range(24):
            perm_mat = self.get_permutation_matrix(cayley, i)
            w = W[:,:,:,:,:,:,i]
            w = w.permute([0,1,2,3,5,4])
            w = torch.reshape(w, [-1, 24])
            w = torch.matmul(w, perm_mat)
            w = torch.reshape(w, list(Wsh[:4])+[-1,24])
            U.append(w.permute([0,1,2,3,5,4]))
        return U


    def get_permutation_matrix(self, perm, dim):
        ndim = perm.shape[0]
        mat = np.zeros((ndim, ndim))
        for j in range(ndim):
            mat[j,perm[j,dim]] = 1
        return torch.from_numpy(mat).float()


    def get_cayleytable(self):
        Z = self.get_s4mat()
        cayley = []
        for y in Z:
            for z in Z:
                r = z @ y
                for i, el in enumerate(Z):
                    if np.sum(np.square(el - r)) < 1e-6:
                        cayley.append(i)
        cayley = np.stack(cayley)
        return np.reshape(cayley, [24,24])


    def get_s4mat(self):
        Z = []
        for i in range(4):
            # Z_4 rotation about Y
            # S^2 rotation
            for j in range(4):
                z = self.get_3Drotmat(i,j,0)
                Z.append(z)
            # Residual pole rotations
            Z.append(self.get_3Drotmat(i,0,1))
            Z.append(self.get_3Drotmat(i,0,3))
        return Z


    def get_3Drotmat(self,x,y,z):
        c = [1.,0.,-1.,0.]
        s = [0.,1.,0.,-1]

        Rx = np.asarray([[c[x],     -s[x],  0.],
                         [s[x],     c[x],   0.],
                         [0.,       0.,     1.]])
        Ry = np.asarray([[c[y],     0.,     s[y]],
                         [0.,       1.,     0.],
                         [-s[y],    0.,     c[y]]])
        Rz = np.asarray([[1.,       0.,     0.],
                         [0.,       c[z],   -s[z]],
                         [0.,       s[z],   c[z]]])
        return Rz @ Ry @ Rx
