# -*- coding: utf-8 -*-

import numpy as np
from math import pi

class Polyhedron():
    '''
    Class for irregular polyhedron

    Parameters
    ----------
    a:          float, default:0
                Diameter of the inclusion along direction-1 in voxel units
    b:          float, default: b=a
                Diameter of the inclusion along direction-2 in voxel units
    c:          float, default: c=a
                Diameter of the inclusion along direction-3 in voxel units
    n_cuts:     int, default:10
                Number of faces for the irregular polyhedron.

    '''
    
    def __init__(self, a, b, c, n_cuts=10):
        self.a = a
        self.b = b
        self.c = c
        self.n_cuts = n_cuts
        
        dia = max(self.a, self.b, self.c)
        rad = dia / 2
        if (dia) % (round(dia)) != 0:
            dia = round(dia)
        if rad % (round(rad)) != 0:
            dia = dia + 1
            
        self.dia = dia
        self.vol_est = ((dia + 1) ** 3 + (1.0 / 6.0 * pi * a * b * c)) / 2
        self.mat_inc = np.zeros((int(self.dia + 1), int(self.dia + 1), int(self.dia + 1))).astype(int)
        
        self.vox_inc = int(1)
    
    def generate_inclusion_matrix(self):
        '''
        generate polyhedron for the given size. The polyhedron will have
        major axis -size and minor axes as size*aspRat_1 and size*aspRat_2.

        Return
        ------
        mat_inc:    3D array (int/bool)
                    3D voxel representation of ellipsoid'''

        a = np.copy(self.a)
        b = np.copy(self.b)
        c = np.copy(self.c)
        rx = a/2.0; ry = b/2.0; rz = c/2.0
        n_cuts = self.n_cuts
        theta_faces = np.random.random((3, self.n_cuts)) * 2 * pi
        theta_inc = np.random.random((3)) * 2 * pi
        vox = self.vox_inc
        
        dia = np.array(np.shape(self.mat_inc))
        u = np.array([1, 0, 0])
        r_mat = np.array([[rx, 0, 0], [0, ry, 0], [0, 0, rz]])
        del_el = lambda x, y, z: np.array([(2.0 / rx ** 2) * x, (2.0 / ry ** 2) * y, (2.0 / rz ** 2) * z])
        qx_rot, qy_rot, qz_rot = self.__get_rotation_matrix(theta_inc[2:3], theta_inc[1:2], theta_inc[0:1])
        p_rot = np.copy(qx_rot[0, :, :].dot((qy_rot[0, :, :]).dot(qz_rot[0, :, :])))
        qx_rot, qy_rot, qz_rot = self.__get_rotation_matrix(theta_faces[2, :], theta_faces[1, :], theta_faces[0, :])
        x_vector = np.einsum('ab, ibc, icd, ide, e->ai', r_mat, qx_rot, qy_rot, qz_rot, u, optimize='greedy')
        delF = np.transpose(np.einsum('ab, bi->ai', p_rot, del_el(x_vector[0, :], x_vector[1, :], x_vector[2, :])))
        idd = (dia - 1) / 2
        coords = np.meshgrid(np.arange(0, dia[0]), np.arange(0, dia[1]), np.arange(0, dia[2]))
        [x, y, z] = [coords[0].ravel().astype(int), coords[1].ravel().astype(int), coords[2].ravel().astype(int)]
        points = np.transpose(np.array([coords[0].ravel(), coords[1].ravel(), coords[2].ravel()]).astype(float)) - idd
        check = np.einsum('ij, kj->ik', delF, points) <= 2
        inside = np.sum(check, 0) == n_cuts
        self.mat_inc[x[inside], y[inside], z[inside]] = int(vox)
        return self.mat_inc
        
        
    def __get_rotation_matrix(self, th1, th2, th3):
        '''
        Compute rotation matrix for the given angles th1, th2, th3.

        Parameters
        ----------
        th1:     array of size N of angles about axis-1. Values between 0 to 2pi.
        th2:     array of size N of angles about axis-2. Values between 0 to 2pi.
        th3:     array of size N of angles about axis-3. Values between 0 to 2pi.
        
        Return
        ------
        qx_rpt: array of size 3X3XN, type: float
                3D rotation angle about axis-1 for all given angles th1.
        qy_rot: array of size 3X3XN, type: float
                3D rotation angle about axis-2 for all given angles th2.
        qz_rot: array of size 3X3XN, type: float
                3D rotation angle about axis-3 for all given angles th3.  
        '''
        qz_rot = np.transpose(np.array([[np.cos(th1), -np.sin(th1), np.zeros((np.size(th1)))],
                                     [np.sin(th1), np.cos(th1), np.zeros((np.size(th1)))],
                                     [np.zeros((np.size(th1))), np.zeros((np.size(th1))), np.ones((np.size(th1)))]]),
                           (2, 0, 1))
        qy_rot = np.transpose(np.array([[np.cos(th2), np.zeros((np.size(th2))), np.sin(th2)],
                                     [np.zeros((np.size(th2))), np.ones((np.size(th2))), np.zeros((np.size(th2)))],
                                     [-np.sin(th2), np.zeros((np.size(th2))), np.cos(th2)]]), (2, 0, 1))
        qx_rot = np.transpose(np.array([[np.ones(np.size(th3)), np.zeros((np.size(th3))), np.zeros((np.size(th3)))],
                                     [np.zeros((np.size(th3))), np.cos(th3), -np.sin(th3)],
                                     [np.zeros((np.size(th3))), np.sin(th3), np.cos(th3)]]), (2, 0, 1))

        return qx_rot, qy_rot, qz_rot        
    
        
    def generate_new_inclusion(self):
        '''
        Instantiate new polyhedron object and generates the matrix
        '''
        inclusion = Polyhedron(a=self.a, b=self.b, c=self.c, n_cuts=self.n_cuts, vox_inc=self.vox_inc)
        inclusion.generate_inclusion_matrix()
        inclusion.compute_vox_volume()
        return inclusion
    
    def compute_vox_volume(self):
        '''  Compute voxel volume of the inclusion  '''
        self.vol_vox = np.sum(self.mat_inc == self.vox_inc)
        
        
        
        
        
    