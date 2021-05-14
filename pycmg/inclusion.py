# -*- coding: utf-8 -*-

import numpy as np
from math import pi

class Polyhedron():
    '''
    Class for irregular polyhedron

    Parameters
    ----------
    a:          float, default:0
                Radius of the inclusion along direction-1
    b:          float, default: b=a
                Radius of the inclusion along direction-2
    c:          float, default: c=a
                Radius of the inclusion along direction-3
    n_cuts:     int, default:20
                Number of faces for the irregular polyhedron.
    coat:       bool, default:False
                Coating on inclusion, True/False.
    t_coat:     float, default:0
                Thickness of coating on the polyhedron.
    space:      bool, default:False
                Space (like coating) on the polyhedron, True/False.
                This spacing creates gap between inclusion when assembled in
                the main meso/microstructure.
    t_space:    float, default:0
                Thickness of spacing on the polyhedron.
    concave:    bool, default: False
                Provision to apply coating on the polyhedron, True/False.
    n_concave:  int, default:0
                Number of concave depressions on the polyhedron surface.
    depth:      float, default:0
                Parameter to determine depth of concave depression on the
                polyhedron surface (from 0 to 1)
    width:      float, default:0
                Parameter to determine width of concave depression on the

    '''

    def __init__(self, a=0, b=None, c=None, coat=False, t_coat=0,
                 space=False, t_space=0, n_cuts=20, concave=False, n_concave=0, depth=0, width=0, vox_inc=1, vox_coat=2, vox_space=3):

        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.n_cuts = int(n_cuts)
        self.n_concave = int(n_concave)
        self.concave = concave
        self.depth = float(depth)
        self.width = float(width)

        if self.b is None:
            self.b = self.a
        if self.c is None:
            self.c = self.a

        dia = 2 * max(self.a, self.b, self.c)
        if (dia) % (round(dia)) != 0:
            dia = round(dia)

        rad = dia / 2
        if rad % (round(rad)) != 0:
            dia = dia + 1

        self.dia = dia
        self.vol_est = ((dia + 1) ** 3 + (4.0 / 3.0 * pi * a * b * c)) / 2
        self.mat_inc = np.zeros((int(self.dia + 1), int(self.dia + 1), int(self.dia + 1))).astype(int)
        self.coat = coat
        self.t_coat = float(t_coat)
        self.space = space
        self.t_space = float(t_space)
        self.vox_inc = int(vox_inc)
        self.vox_space = int(vox_space)
        self.vox_coat = int(vox_coat)

    def generate_polyhedron(self, a, b, c, n_cuts, theta_faces, theta_inc,
                            vox):
        '''
        generate polyhedron for the given size. The polyhedron will have
        major axis -size and minor axes as size*aspRat_1 and size*aspRat_2.

        Parameters
        ----------
        a:          float
                    Major/minor axis 1.
        b:          float, default:b=a
                    Major/minor axis 2.
        c:          float, default:c=a
                    Major/minor axis 3
        n_cuts:     int
                    Number of faces for the polyhedron.
        theta_faces:array of size (3,n_cuts), type float between 0 to 2*pi
                    Polyhedron faces angles
        theta_inc: array of size 3, type float between 0 to 2*pi
                    3D angle of the ellipsoid.
        vox:        int
                    Voxel value to represent ellipsoid shape.

        Return
        ------
        mat_inc:    3D array (int/bool)
                    3D voxel representation of ellipsoid'''

        if int(n_cuts) != np.shape(theta_faces)[1]:
            raise Exception('Provide angles for all polyhedron faces')

        dia = np.array(np.shape(self.mat_inc))
        u = np.array([1, 0, 0])
        R = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
        delEl = lambda x, y, z: np.array([(2.0 / a ** 2) * x, (2.0 / b ** 2) * y, (2.0 / c ** 2) * z])
        Qx1, Qy1, Qz1 = self.getRotationMatrix(theta_inc[2:3], theta_inc[1:2], theta_inc[0:1])
        P = np.copy(Qx1[0, :, :].dot((Qy1[0, :, :]).dot(Qz1[0, :, :])))
        Qx1, Qy1, Qz1 = self.getRotationMatrix(theta_faces[2, :], theta_faces[1, :], theta_faces[0, :])
        X = np.einsum('ab, ibc, icd, ide, e->ai', R, Qx1, Qy1, Qz1, u, optimize='greedy')
        delF = np.transpose(np.einsum('ab, bi->ai', P, delEl(X[0, :], X[1, :], X[2, :])))
        idd = (dia - 1) / 2
        coords = np.meshgrid(np.arange(0, dia[0]), np.arange(0, dia[1]), np.arange(0, dia[2]))
        [x, y, z] = [coords[0].ravel().astype(int), coords[1].ravel().astype(int), coords[2].ravel().astype(int)]
        points = np.transpose(np.array([coords[0].ravel(), coords[1].ravel(), coords[2].ravel()]).astype(float)) - idd
        check = np.einsum('ij, kj->ik', delF, points) <= 2
        inside = np.sum(check, 0) == n_cuts
        self.mat_inc[x[inside], y[inside], z[inside]] = int(vox)
        return self.mat_inc, delF

    def apply_coating(self, mat_inc, a, b, c, delF, n_cut, t_coat, vox):
        ''' 
        Apply coating/spacing on the polyhedron surface. Spacing is equivalent to applying coating on top of the inclusion and it ensures gap between aggregates when assembled in the micro/mesostructure.

        Parameters
        ----------
        mat_inc:        3D array of type int
                        Provides voxel representation of the polyhedron with/without coating and spacing
        a:              float
                        Major/minor axis 1 of the polyhedron.
        b:              float, default:b=a
                        Major/minor axis 2 of the polyhedron.
        c:              float, default:c=a
                        Major/minor axis 3 of the polyhedron.
        delF:           array of type float
                        Gradient vector for all polyhedron planes.
        n_cut:          int
                        Number of polyhedron planes.
        t_coat:         float
                        Thickness of the coating/spacing on top of the inclusion/polyhedron
        vox:            int
                        Voxel value to represent coating / spacing
                        
        Return
        ------
        I_coat:         3D array of type int
                        New voxel representation of the polyhedron with coat/spacing.
        delF:           array of type float
                        Updated gradient vector with coat.
        a,b,c:          float
                        New raidus of the polyhedron with coat/spacing.
        
        '''
        t_coat = int(round(t_coat));
        dia = np.array(np.shape(mat_inc)) + (2 * int(np.round(t_coat)))
        mat_coat = np.zeros((dia))
        mat_coat[t_coat:dia[0] - t_coat, t_coat:dia[1] - t_coat, t_coat:dia[2] - t_coat] = mat_inc
        rat_coat = (max(a, b, c) + t_coat) / max(a, b, c)
        idd = (dia - 1) / 2
        coords = np.meshgrid(np.arange(0, dia[0]), np.arange(0, dia[1]), np.arange(0, dia[2]))
        [x, y, z] = [coords[0].ravel().astype(int), coords[1].ravel().astype(int), coords[2].ravel().astype(int)]
        points = np.transpose(np.array([coords[0].ravel(), coords[1].ravel(), coords[2].ravel()]).astype(float)) - idd
        delF = delF / rat_coat
        check_coat = np.einsum('ij, kj->ik', delF, points) <= 2
        inside = np.logical_and(mat_coat[x, y, z] == 0, np.sum(check_coat, 0) == n_cut)
        mat_coat[x[inside], y[inside], z[inside]] = int(vox)
        a, b, c = a + t_coat, b + t_coat, c + t_coat
        return mat_coat, delF, a, b, c

    def generate_inclusion_matrix(self):
        '''
        generate polyhedron inclusion
        '''

        N = self.n_cuts
        theta = np.random.random((3, self.n_cuts)) * 2 * pi
        self.theta = theta
        beta = np.random.random((3)) * 2 * pi
        self.beta = beta
        a = self.a;
        b = self.b;
        c = self.c
        mat_inc, delF = self.generate_polyhedron(a, b, c, N, theta, beta, self.vox_inc)
        a, b, c = np.copy(self.a), np.copy(self.b), np.copy(self.c)
        if self.concave is True:
            eta = np.random.random((3, self.n_concave)) * 2 * pi
            if self.coat is True:
                if self.space is True:
                    mat_inc = self.generate_concave_depression(mat_inc, a, b, c, self.n_concave, self.depth, self.width,
                                                               eta, beta)
                    mat_inc, delF, a, b, c = self.apply_coating(mat_inc, a, b, c, delF, N, self.t_coat, self.vox_coat)
                    mat_inc = self.generate_concave_depression(mat_inc, a, b, c, self.n_concave, self.depth, self.width,
                                                               eta, beta)
                    mat_inc, delF, a, b, c = self.apply_coating(mat_inc, a, b, c, delF, N, self.t_coat, self.vox_space)
                    mat_inc = self.generate_concave_depression(mat_inc, a, b, c, self.n_concave, self.depth, self.width,
                                                               eta, beta)

                else:
                    mat_inc = self.generate_concave_depression(mat_inc, a, b, c, self.n_concave, self.depth, self.width,
                                                               eta, beta)
                    mat_inc, delF, a, b, c = self.apply_coating(mat_inc, a, b, c, delF, N, self.t_coat, self.vox_coat)
                    mat_inc = self.generate_concave_depression(mat_inc, a, b, c, self.n_concave, self.depth, self.width,
                                                               eta, beta)

            else:
                if self.space is True:
                    mat_inc = self.generate_concave_depression(mat_inc, a, b, c, self.n_concave, self.depth, self.width,
                                                               eta, beta)
                    mat_inc, delF, a, b, c = self.apply_coating(mat_inc, a, b, c, delF, N, self.t_coat, self.vox_space)
                    mat_inc = self.generate_concave_depression(mat_inc, a, b, c, self.n_concave, self.depth, self.width,
                                                               eta, beta)

                else:
                    mat_inc = self.generate_concave_depression(mat_inc, a, b, c, self.n_concave, self.depth, self.width,
                                                               eta, beta)

        else:
            if self.coat is True:
                if self.space is True:
                    mat_inc, delF, a, b, c = self.apply_coating(mat_inc, a, b, c, delF, N, self.t_coat, self.vox_coat)
                    mat_inc, delF, a, b, c = self.apply_coating(mat_inc, a, b, c, delF, N, self.t_coat, self.vox_space)
                else:
                    mat_inc, delF, a, b, c = self.apply_coating(mat_inc, a, b, c, delF, N, self.t_coat, self.vox_coat)
            else:
                if self.space is True:
                    mat_inc, delF, a, b, c = self.apply_coating(mat_inc, a, b, c, delF, N, self.t_space, self.vox_space)

        self.mat_inc = mat_inc
        return mat_inc

    def generate_new_inclusion(self):
        '''
        Instantiate new polyhedron object and generates the matrix
        '''
        inclusion = Polyhedron(a=self.a, b=self.b, c=self.c, coat=self.coat, t_coat=self.t_coat,
                               space=self.space, t_space=self.t_space, vox_inc=self.vox_inc, vox_coat=self.vox_coat,
                               vox_space=self.vox_space,
                               n_cuts=self.n_cuts, concave=self.concave, n_concave=self.n_concave, depth=self.depth,
                               width=self.width, x0=self.x0)
        inclusion.generate_inclusion_matrix()
        inclusion.compute_vox_volume()
        return inclusion
    
    def compute_vox_volume(self):
        '''  Compute voxel volume of the inclusion  '''
        self.vol_vox = np.sum(self.mat_inc == self.vox_inc) + np.sum(self.mat_inc == self.vox_coat)

    def getRotationMatrix(self, th1, th2, th3):
        '''
        Compute rotation matrix for the given angles th1, th2, th3.

        Parameters
        ----------
        th1:     array of size N of angles about axis-1. Values between 0 to 2pi.
        th2:     array of size N of angles about axis-2. Values between 0 to 2pi.
        th3:     array of size N of angles about axis-3. Values between 0 to 2pi.
        
        Return
        ------
        Qx1:    array of size 3X3XN, type: float
                3D rotation angle about axis-1 for all given angles th1.
        Qy1:    array of size 3X3XN, type: float
                3D rotation angle about axis-2 for all given angles th2.
        Qz1:    array of size 3X3XN, type: float
                3D rotation angle about axis-3 for all given angles th3.  
        '''
        Qz1 = np.transpose(np.array([[np.cos(th1), -np.sin(th1), np.zeros((np.size(th1)))],
                                     [np.sin(th1), np.cos(th1), np.zeros((np.size(th1)))],
                                     [np.zeros((np.size(th1))), np.zeros((np.size(th1))), np.ones((np.size(th1)))]]),
                           (2, 0, 1))
        Qy1 = np.transpose(np.array([[np.cos(th2), np.zeros((np.size(th2))), np.sin(th2)],
                                     [np.zeros((np.size(th2))), np.ones((np.size(th2))), np.zeros((np.size(th2)))],
                                     [-np.sin(th2), np.zeros((np.size(th2))), np.cos(th2)]]), (2, 0, 1))
        Qx1 = np.transpose(np.array([[np.ones(np.size(th3)), np.zeros((np.size(th3))), np.zeros((np.size(th3)))],
                                     [np.zeros((np.size(th3))), np.cos(th3), -np.sin(th3)],
                                     [np.zeros((np.size(th3))), np.sin(th3), np.cos(th3)]]), (2, 0, 1))

        return Qx1, Qy1, Qz1

    def generate_concave_depression(self, mat_inc, a, b, c, n_concave, depth, width, theta_concave, theta_inc):
        '''
        Generate concave depression on the surface of the polyhedron

        Parameters
        ----------
        mat_inc:        3D array of type int
                        Provides voxel representation of the polyhedron
                        with/without coating and spacing
        a:              float
                        Radius of the inclusion along direction-1
                        For cylinder, it is the radius of the cross-section along direction-1
        b:              float
                        Radius of the inclusion along direction-2
                        For cylinder, it is the radius of the cross-section along direction-2
        c:              float
                        Radius of the inclusion along direction-3
                        For cylinder, it is the half of the length
        n_concave:      int
                        Number of concave depressions on the polyhedron surface.
        depth:          float
                        Parameter determining depth of the concave depressions
                        (value between 0 to 1).
        width:          float
                        Parameter determining width of the concave depressions.
        theta_conave:   array of size (3,n_concave), type float between 0 to
                        2*pi
                        Angle of concave depressions on the polyhedron surface.
        theta_inc:      array of size (3), type float between 0 to 2*pi
                        Polyhedron 3D angle
                        
        Return
        ------
        mat_inc:        3D array of type int
                        New voxel representation of the polyhedron with concave depressions. 
        '''
        dia = np.shape(mat_inc)

        u = np.array([1, 0, 0]);
        R = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
        Qx1, Qy1, Qz1 = self.getRotationMatrix(theta_inc[2:3], theta_inc[1:2], theta_inc[0:1])
        P = np.copy(Qx1[0, :, :].dot((Qy1[0, :, :]).dot(Qz1[0, :, :])))
        Qx1, Qy1, Qz1 = self.getRotationMatrix(theta_concave[2, :], theta_concave[1, :], theta_concave[0, :])
        Xnorm = np.sqrt(
            np.sum(np.square(np.einsum('ab, bc, icd, ide, ief, f->ai', P, R, Qx1, Qy1, Qz1, u, optimize='greedy')), 0))
        idd = (np.array(dia) - 1) / 2
        coords = np.meshgrid(np.arange(0, dia[0]), np.arange(0, dia[1]), np.arange(0, dia[2]))
        [x, y, z] = [coords[0].ravel().astype(int), coords[1].ravel().astype(int), coords[2].ravel().astype(int)]
        points = np.transpose(np.array([coords[0].ravel(), coords[1].ravel(), coords[2].ravel()]).astype(float)) - idd

        if width == 0 or depth == 0:
            raise Exception('concave depth and width values cannot be zero')

        kc = np.einsum('ma, iab, ibc, icd, jd->imj', P, Qx1, Qy1, Qz1, points, optimize='greedy')
        g1 = 0.1 * (kc[:, 1, :] ** 2) / width
        g2 = 0.1 * (kc[:, 2, :] ** 2) / width
        g3 = kc[:, 0, :]
        gauss = np.einsum('i, ia->ia', -depth * Xnorm.ravel(), np.exp(-(g1 + g2)), optimize='greedy') - g3
        check_concave = gauss > (-Xnorm.ravel())[:, None]
        outside = np.logical_and(mat_inc[x, y, z] != 0, np.sum(check_concave, 0) != n_concave)
        mat_inc[x[outside], y[outside], z[outside]] = 0
        return mat_inc