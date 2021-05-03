import os
import sys
import time

import numpy as np

import torch
import torchvision.transforms.functional as torchtransforms

class T4_group(object):
    def __init__(self):
        self.group_dim = 12
        self.cayleytable = self.get_cayleytable()

    
    def rotate(self, x, axis, shift):
        angles = [0.,np.pi/2.,np.pi,3.*np.pi/2.]
        perm = ([0,3,2,1],[0,1,3,2],[0,2,1,3])
        x = x.permute(perm[axis])
        x = torchtransforms.rotate(x, angles[shift])
        return x.permute(perm[axis])


    def r1(self, x):
        x = self.rotate(x, 0, -1)
        return self.rotate(x, 1, 1)

        
    def r2(self, x):
        x = self.rotate(x, 0, -1)
        return self.rotate(x, 1, -1)


    def r3(self, x):
        return self.rotate(x, 0, 2)


    def get_Grotations(self, x):
        """Rotate the tensor x with all 12 T4 rotations

        Args:
            x: [n_channels, h,w,d]
        Returns:
            list of 12 rotations of x [[n_channels,h,w,d],....]
        """
        Z = []
        for i in range(3):
            y = x
            for __ in range(i):
                y = self.r1(y) 
            for j in range(3):
                z = y
                for __ in range(j):
                    z = self.r2(z)
                Z.append(z)
        for i in range(3):
            z = self.r3(x)
            for __ in range(i):
                z = self.r2(z) 
            Z.append(z)
        return Z


    def G_permutation(self, W):
        """Permute the outputs of the group convolution
        
        Args:
            W: [n_channels_in, h, w, d, group_dim, n_channels_out, group_dim]
        Returns:
            list of 12 permutations of W [[n_channels_in, h, w, d, group_dim, n_channels_out, group_dim],....]
        """
        Wsh = W.shape
        cayley = self.cayleytable
        U = []
        for i in range(12):
            perm_mat = self.get_permutation_matrix(cayley, i)
            w = W[:,:,:,:,:,:,i]
            w = w.permute([0,1,2,3,5,4])
            w = w.view([-1, 12])
            w = torch.matmul(w, perm_mat)
            w = w.view(list(Wsh[:4])+[-1,12])
            U.append(w.permute([0,1,2,3,5,4]))
        return U


    def get_permutation_matrix(self, perm, dim):
        """Creates and return the permutation matrix
        
        Args:
            perm: numpy matrix (Cayley matrix of the group)
        Returns:
            float Tensor
        """
        # TODO : make cleaner
        ndim = perm.shape[0]
        mat = np.zeros((ndim, ndim))
        for j in range(ndim):
            mat[j,perm[j,dim]] = 1
        return torch.from_numpy(mat).float()


    def get_t4mat(self):
        Z = []
        for i in range(3):
            y = np.eye(3)
            for __ in range(i):
                y = y @ self.get_3Drotmat(1,0,0)   
                y = y @ self.get_3Drotmat(0,-1,0)  
            for j in range(3):
                z = y
                for __ in range(j):
                    #z = r2(z)
                    z = z @ self.get_3Drotmat(1,0,0)  
                    z = z @ self.get_3Drotmat(0,1,0) 
                Z.append(z)
        for i in range(3):
            #z = r3(x)
            z = self.get_3Drotmat(2,0,0) 
            for __ in range(i):
                #z = r2(z) 
                z = z @ self.get_3Drotmat(1,0,0) 
                z = z @ self.get_3Drotmat(0,1,0)  
            Z.append(z)
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


    def get_cayleytable(self):
        """Returns the Cayley table of V group

        Returns:
            4 by 4 numpy array
        """
        Z = self.get_t4mat()
        cayley = []
        for y in Z:
            for z in Z:
                r = z @ y
                for i, el in enumerate(Z):
                    if np.sum(np.square(el - r)) < 1e-6:
                        cayley.append(i)
        cayley = np.stack(cayley)
        return np.reshape(cayley, [12,12]).T


