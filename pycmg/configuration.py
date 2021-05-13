# -*- coding: utf-8 -*-

import numpy as np
from inclusion import Polyhedron, Ellipsoid, Cylinder
import pandas as pd


class Configuration():

    '''
    Provides methods for configuring the geometrical and topological parameters of the mesostructure.
    '''

    def __init__(self,vf_max_assembly=0.3):
        self.inclusionFamList = []
        self.inclusionFamIdList = []
        self.inclusionVolumeList = []
        self.inclusionSizeList = []
        self.inclusionFamId_count = 0
        self.vf_max_assembly = vf_max_assembly

    def load_inclusions(self, conf_csv=None, conf_header=[],
                        conf_values=[], conf_dict=None):

        '''
        :param conf_csv: string (with .csv extension), Location of the csv file which has aggregate parameters.

        .. csv-table:: An example csv file for concrete
           :file: ../tutorials/tutorial_3/AB8_gv2_test.csv
           :header-rows: 1
           :class: longtable
           :widths: 1,1,1,1,1,1,1,1,1,1,1,1,1

        .. note::
          **The header of the parameters in the csv file should be as follows:**

          - inclusion_type: Aggregate type such as ('Sphere,Polyhedron,Ellipsoid and Cylinder etc.')
          - a:              radius of the inclusion along direction-1. For cylinder, it is the radius of the cross-section along direction-1
          - b:              radius of the inclusion along direction-2. For cylinder, it is the radius of the cross-section along direction-2
          - c:              radius of the inclusion along direction-3. For cylinder, it is the half of the length
          - n_cuts:         Number of faces/cuts for the polyhedron shaped aggregates (not applicable for other shapes).
          - concave:        Yes/No. Provision for concave depressions on the aggregates (not applicable for cylindrical shaped aggregates)
          - n_concave:      Number of concave depressions on each aggregate surface (not applicable for cylindrical shaped aggregates).
          - depth:          A parameter which determines depth of the concave depression on the aggregate surface (not applicable for cylindrical shaped aggregates). Values should be between 0 to 1 (0 lowest, 1 highest).
          - width:          A parameter which determines width of the concave depression on the aggregate surface (not applicable for cylinderical shaped aggregates).
          - coat:           Yes/No. Provision for the coating on the aggregate surface.
          - t_coat:         Thickness of the coating on the aggregate surface.
          - space:          Yes/No. Provision for the spacing on the aggregate surface. Spacing is like a coat on top of the aggregate which provides minimum gap between each inclusion in the mesostructure.
          - t_space:        Thickness of the spacing on the aggregate surface.
          - vox_inc:        Voxel value for the aggregate.
          - vox_coat:       Voxel value for the coating.
          - vf_max:         Maximum volume fraction of each sized aggregates (value between 0 to 1).

        :param conf_header: If not csv, then a header with parameter names as given above and corresponding array of values have to be loaded.
        :param conf_values: In not csv, then values corresponding to the header have to be loaded.
        :param conf_dict:   If not csv and conf_header & conf_values, inputs can be also given through a dictionary.
        '''

        if conf_csv is not None:
            data = pd.read_csv(conf_csv)
            if len(conf_header) == 0:
                conf_header = data.columns
            if len(conf_values) != 0:
                raise Exception('csv file has already been loaded')
                
            if conf_dict is not None:
                raise Exception('csv file has already been loaded')
            conf_values = np.array(data)
            data.replace(r'^\s*$', np.nan, regex=True)
            if data.isnull().values.any() is True:
                raise Exception('csv file has empty cells')

            if len(conf_header) != np.shape(conf_values)[1]:
                raise Exception('header length and values length are different!')
        elif conf_dict is not None:
            conf_header, conf_values = zip(*conf_dict.items())
            conf_values = np.transpose(np.array(conf_values))
            
        elif len(conf_header)==0:
            print('No inputs for the inclusions are given. Hence default inclusion (sphere of size 10) is created')
            conf_header=['inclusion_type','a']  
            conf_values=['Ellipsoid',10] 
            
        else:
            conf_values = np.transpose(np.array(conf_values))
        if np.size(np.shape(conf_values)) == 1:
            conf_values = np.reshape(conf_values, (1, np.shape(conf_values)[0]))
            
        if len(conf_header) != np.shape(conf_values)[1]:
                raise Exception('header length and values length are different!')
        conf_values=np.array(conf_values)
        for i in range(np.shape(conf_values)[0]):
            aggr = InclusionFamily(kwargs=dict(zip(conf_header, conf_values[i, :])))
            self.inclusionFamIdList.append(self.inclusionFamId_count)
            self.inclusionSizeList.append(max(aggr.standard_inclusion.a,aggr.standard_inclusion.b,aggr.standard_inclusion.c))
            self.inclusionVolumeList.append(aggr.standard_inclusion.vol_vox)
            self.inclusionFamList.append(aggr)
            self.inclusionFamId_count += 1

        self.inclusion_sorted = np.array(self.inclusionFamIdList)

    def sort_inclusions(self, sort_type='size'):
        '''
        Sort aggregates in descending order according to the given sort_type
        Parameters

        :param sort_type: string either 'volume' or 'size', defaults to 'size'
        :type sort_type: string, optional

        '''

        if sort_type == 'volume':
            sort = np.array(self.inclusionVolumeList).argsort()
            inclusion_sorted = np.array(self.inclusionFamIdList)[sort[::-1]]

        if sort_type == 'size':
            sort = np.array(self.inclusionSizeList).argsort()
            inclusion_sorted = np.array(self.inclusionFamIdList)[sort[::-1]]

        self.inclusion_sorted = inclusion_sorted


class InclusionFamily():
    '''
    This class is for family of inclusions
    
    Parameters
    ----------
    inclusion_type:   string, default: None
                      Gives the type of inclsion (Polyhedron/Sphere/Ellipsoid/Cylinder).
    Id:               int, default: None
                      Id of the inclusion family.
    inclusionList:    array/list (1D)
                      Gives list of inclusions belonging to the current family.
    vf_max:           float, value between 0 to 1, default:1.0
                      Maximum volume fraction of the inclusion family.
    a:                float, default:10
                      Radius of the inclusion along direction-1
                      For cylinder it is the radius of the cross-section along direction-1.
    b:                Radius of the inclusion along direction-2
                      For cylinder it is the radius of the cross-section along direction-2.
    c:                Radius of the inclusion along direction-3
                      For cylinder it is the half of the length.
    n_cuts:           int, default:10
                      Number of faces of the irregular polyhedron. Doesnt apply for ellipsoid/sphere/cylinder.
    concave:          bool, True/False, default:False
                      Boolean for concave depression on inclusion surface. Not applicable for cylinder shape.
    n_concave:        int, default:0
                      Number of concave depressions on the inclusion surface. Not applicable for cylinder shape.
    depth:            float, value between 0 to 1, default:0
                      Parameter which determines depth of the concave depression from the inclusion surface.
                      Not applicable for cylinder shape.
    width:            float, default:0
                      Parameter which determines width of the concave depression on the inclusion surface.
                      Not applicable for cylinder shape.
    coat:             bool, True/False, default:False
                      Boolean for coat on inclusion. 
    t_coat:           float, default:0
                      Thickness of the coating.
    space:            bool, True/False, default:False
                      Boolean for space which determines gap between inclusions in micro/mesostructure.
    t_space:          flaot, default:0
                      Thickness of the spacing.
    vox_inc:          int, default:1
                      Voxel value for the inclusion.
    vox_coat:         int, default:2
                      Voxel value for the coat.
    x:                float, default:0
                      Location along x direction to assemble the inclusions belonging to this family.
    y:                float, default:0
                      Location along y direction to assemble the inclusions belonging to this family.
    z:                float, default:0
                      Location along z direction to assemble the inclusions belonging to this family.
    kwargs:           Other parameters, default:None
    
    '''
    def __init__(self, inclusion_type=None, Id=None, inclusionList=[],
                 vf_max=1, a=10, b=0, c=0, n_cuts=10, concave=False, n_concave=0,
                 depth=0, width=0, coat=False,
                 t_coat=0, space=False, t_space=0,
                 vox_inc=1, vox_coat=2, x=0, y=0, z=0,
                 kwargs=None):
        self.inclusion_type = inclusion_type
        self.inclusionList = inclusionList
        self.vf_max = vf_max
        self.a = a
        self.b = b
        self.c = c
        self.n_cuts = n_cuts
        self.concave = concave
        self.n_concave = n_concave
        self.depth = depth
        self.width = width
        self.coat = coat
        self.t_coat = t_coat
        self.space = space
        self.t_space = t_space
        self.vox_inc = vox_inc
        self.vox_coat = vox_coat
        self.vox_space = 100
        self.vf_max = vf_max
        self.standard_inclusion = None
        self.Id = None
        self.vf = 0
        self.x = x
        self.y = y
        self.z = z

        self.__dict__.update(kwargs)

        self.a = float(self.a)
        self.b = float(self.b)
        self.c = float(self.c)
        if self.b == 0:
            self.b = self.a
        if self.c == 0:
            self.c = self.b
        self.vf_max = float(self.vf_max)
        self.n_cuts = int(self.n_cuts)
        self.concave = self.concave
        self.n_concave = int(self.n_concave)
        self.depth = float(self.depth)
        self.width = float(self.width)
        self.coat = self.coat
        self.t_coat = float(self.t_coat)
        self.space = self.space
        self.t_space = float(self.t_space)
        self.vox_inc = int(self.vox_inc)
        self.vox_coat = int(self.vox_coat)
        self.vox_space = int(self.vox_space)
        self.vf_max = float(self.vf_max)
        self.x = int(round(float(self.x)))
        self.y = int(round(float(self.y)))
        self.z = int(round(float(self.z)))
        self.x0 = np.array([self.x, self.y, self.z])
        self.standard_inclusion = self.generate_inclusion()
        if self.vox_space != 100:
            raise Exception('voxel value for spacing should be 100')

    def generate_inclusion(self):
        '''
        This method generates an inclusion

        :return: Object of class :py:class:`smg_inclusion.Inclusion`.
        '''

        if self.inclusion_type == 'Polyhedron':
            inclusion = Polyhedron(a=self.a, b=self.b, c=self.c, coat=self.coat,
                                            t_coat=self.t_coat, space=self.space,
                                               t_space=self.t_space, vox_inc=self.vox_inc,
                                               vox_coat=self.vox_coat, vox_space=self.vox_space, 
                                               n_cuts=self.n_cuts, concave=self.concave, 
                                               n_concave=self.n_concave,
                                               depth=self.depth, width=self.width, x0=self.x0)
        elif self.inclusion_type == 'Ellipsoid' or self.inclusion_type == 'Sphere':
            inclusion = Ellipsoid(a=self.a, b=self.b, c=self.c, coat=self.coat,
                                            t_coat=self.t_coat, space=self.space,
                                               t_space=self.t_space, vox_inc=self.vox_inc,
                                               vox_coat=self.vox_coat, vox_space=self.vox_space, 
                                               concave=self.concave, n_concave=self.n_concave,
                                               depth=self.depth, width=self.width, x0=self.x0)
            
        elif self.inclusion_type == 'Cylinder':
            inclusion = Cylinder(a=self.a, b=self.b, c=self.c, coat=self.coat,
                                            t_coat=self.t_coat, space=self.space,
                                               t_space=self.t_space, vox_inc=self.vox_inc,
                                               vox_coat=self.vox_coat, vox_space=self.vox_space, x0=self.x0)
            
        else:
            raise Exception(' inclusion type is invalid')
            
        inclusion.generate_inclusion_matrix()
        inclusion.compute_vox_volume()
        return inclusion

