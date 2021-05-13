# -*- coding: utf-8 -*-


import numpy as np
import os
import pandas as pd


def export_data(data, export_type='vtk', fileName='mesostructure.vti'):
    '''
    The function exports data in the 3D array to the given export type (ex. vtk, npy, npz, csv, txt etc.).
    
    Parameters
    ----------
    data:     3D array of size NXNXN, type int
              Micro/Mesostructure/Inclusion in voxel format.
    export_type: str.
             Export type (vtk/csv/txt/npy/npz)
    fileName: str.
             File location and file name with proper extension (./.../fileName.csv for export_type='csv')

    '''

    if not isinstance(data, np.ndarray):
        raise Exception('given data must be ndarray type')

    if export_type == 'vtk':
        ext = os.path.splitext(fileName)

        if ext[1] != '.vti':
            raise Exception('File name extension should be .vti for vtk export type')
        write_vti_format(fileName, data)

    elif export_type == 'csv':
        ext = os.path.splitext(fileName)

        if ext[1] != '.csv':
            raise Exception('File name extension should be .csv for csv export type')

        data_shape = np.shape(data)

        [x, y, z] = np.meshgrid(np.arange(0, data_shape[0]), np.arange(0, data_shape[1]), np.arange(0, data_shape[2]))

        data_array = np.array([x.ravel(), y.ravel(), z.ravel(), data.ravel()])
        data_array = np.transpose(data_array)
        data_frame = pd.DataFrame(data_array, columns=['x', 'y', 'z', 'values'])
        data_frame.to_csv(fileName, index=False)

    elif export_type == 'txt':
        ext = os.path.splitext(fileName)

        if ext[1] != '.txt':
            raise Exception('File name extension should be .txt for txt export type')

        data_shape = np.shape(data)

        [x, y, z] = np.meshgrid(np.arange(0, data_shape[0]), np.arange(0, data_shape[1]), np.arange(0, data_shape[2]))

        data_array = np.array([x.ravel(), y.ravel(), z.ravel(), data.ravel()])
        data_array = np.transpose(data_array)
        data_frame = pd.DataFrame(data_array, columns=['x', 'y', 'z', 'values'])
        data_frame.to_csv(fileName, index=False)

    elif export_type == 'npy':
        ext = os.path.splitext(fileName)

        if ext[1] != '.npy':
            raise Exception('File name extension should be .npy for npy export type')

        np.save(fileName, data)

    elif export_type == 'npz':
        ext = os.path.splitext(fileName)

        if ext[1] != '.npz':
            raise Exception('File name extension should be .npz for npz export type')
        np.savez(fileName, data)


def write_vti_format(filename, phaseScalar):
    '''
    e.g  write_vti_format('model101.vti', mcrt)
    '''
    with open(filename, 'w+') as f:
        f.writelines('<?xml version="1.0"?> \n')
        f.writelines('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
        # get ndim
        ndim = phaseScalar.shape  # xdim,ydim,zdim
        f.writelines(' ' * 1 + '<ImageData WholeExtent="' + str(0) + ' ' + str(ndim[0]) + ' ' + str(0) + ' ' + str(
            ndim[1]) + ' ' + str(0) + ' ' + str(ndim[2]) + '" Origin="0 0 0" Spacing="1 1 1">\n')
        f.writelines(
            ' ' * 2 + '<Piece Extent="' + str(0) + ' ' + str(ndim[0]) + ' ' + str(0) + ' ' + str(ndim[1]) + ' ' + str(
                0) + ' ' + str(ndim[2]) + '">\n')
        f.writelines(' ' * 3 + '<CellData>\n')
        f.writelines(
            ' ' * 4 + '<DataArray type="Int32" NumberOfComponents="1" Name="Material_phases" format="ascii">\n')

        f.writelines(' ' * 5)
        np.transpose(phaseScalar, [2, 1, 0]).tofile(f, sep=" ", format="%d")  # check write array
        f.writelines('\n' + ' ' * 4 + '</DataArray>\n')

        f.writelines(' ' * 3 + '</CellData>\n')
        vti_tail(f)


def vti_tail(f):
    f.writelines(' ' * 2 + '</Piece>\n')
    f.writelines(' ' * 1 + '</ImageData>\n')
    f.writelines('</VTKFile>')


def visualize_sections(matrix, slices=2):
    '''
    Visualize 2D sections of 3D matrix. Given number of sections (slices) are generated in each direction (xy,xz,yz)
    '''
    import matplotlib.pyplot as plt
    step = np.array(np.shape(matrix)).astype(float) / (float(slices + 1))
    for i in range(slices):
        plt.imshow(matrix[int(step[0] * (i + 1)), :, :])
        plt.title('yz section')
        plt.show()

    for i in range(slices):
        plt.imshow(matrix[:, int(step[1] * (i + 1)), :])
        plt.title('xz section')
        plt.show()

    for i in range(slices):
        plt.imshow(matrix[:, :, int(step[2] * (i + 1))])
        plt.title('xy section')
        plt.show()

    plt.show()
