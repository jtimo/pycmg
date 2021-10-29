# -*- coding: utf-8 -*-

import numpy as np
from inclusion import Polyhedron
import pandas as pd


class Configuration:

    '''
    Provides methods for configuring the geometrical and topological parameters of the mesostructure.
    '''

    def __init__(self,vf_max_assembly=0.3, average_shape=False):
        self.inclusion_fam_list = []
        self.inclusion_fam_id_list = []
        self.inclusion_vol_list = []
        self.inclusion_size_list = []
        self.inclusion_fam_id_count = 0
        self.vf_max_assembly = vf_max_assembly
        self.average_shape = average_shape

    def load_inclusions(self, conf_csv=None):

        '''
        :param conf_csv: string (with .csv extension), Location of the csv file which has aggregate parameters.

        .. csv-table:: An example csv file for concrete
           :file: ../examples/AB8_CMG_full.csv
           :header-rows: 1
           :class: longtable
           :widths: 1,1,1,1,1,1,1,1,1,1,1,1

        .. note::
          **The header of the parameters in the csv file should be as follows:**

          - a:              diameter of the inclusion along direction-1 in actual units (mm/cm etc.).
          - b:              diameter of the inclusion along direction-2 in actual units (mm/cm etc.).
          - c:              diameter of the inclusion along direction-3 in actual units (mm/cm etc.).
          - n_cuts:         Number of faces/cuts for the polyhedron shaped aggregates.
          - concave:        Yes/No. Provision for concave depressions on the aggregates.
          - n_concave:      Number of concave depressions on each aggregate surface in actual units (mm/cm etc.).
          - depth:          A parameter which determines depth of the concave depression on the aggregate surface. Values should be between 0 to 1 (0 lowest, 1 highest).
          - width:          A parameter which determines width of the concave depression on the aggregate surface.
          - coat:           Yes/No. Provision for the coating on the aggregate surface.
          - t_coat:         Thickness of the coating on the aggregate surface in actual units (mm/cm etc.).
          - space:          Yes/No. Provision for the spacing on the aggregate surface. Spacing is like a coat on top of the aggregate which provides minimum gap between each inclusion in the mesostructure.
          - t_space:        Thickness of the spacing on the aggregate surface in actual units (mm/cm etc.).
          - vf_max:         Maximum volume fraction of each sized aggregates (value between 0 to 1).

        :param conf_header: If not csv, then a header with parameter names as given above and corresponding array of values have to be loaded.
        :param conf_values: In not csv, then values corresponding to the header have to be loaded.
        :param conf_dict:   If not csv and conf_header & conf_values, inputs can be also given through a dictionary.
        '''

        if conf_csv is None:
            raise Exception('csv file location is not given')
        data = pd.read_csv(conf_csv)                
        conf_values = np.array(data)
        conf_header = data.columns
        data.replace(r'^\s*$', np.nan, regex=True)
        if data.isnull().values.any() is True:
            raise Exception('csv file has empty cells')
        conf_values=np.array(conf_values)
        for i in range(np.shape(conf_values)[0]):
            aggr = InclusionFamily(average_shape=self.average_shape, kwargs=dict(zip(conf_header, conf_values[i, :])))
            self.inclusion_fam_id_list.append(self.inclusion_fam_id_count)
            self.inclusion_size_list.append(max(aggr.a, aggr.b, aggr.c))
            self.inclusion_fam_list.append(aggr)
            self.inclusion_fam_id_count += 1

        self.inclusion_sorted = np.array(self.inclusion_fam_id_list)


class InclusionFamily:
    '''
    This class is for family of inclusions
    
    Parameters
    ----------
    average_shape:    array of size (3),float
                      Aspect ration of the inclusion along all three axes (value between 0-1).
    Id:               int, default: None
                      Id of the inclusion family.
    inclusion_list:   array/list (1D)
                      Gives list of inclusions belonging to the current family.
    vf_max:           float, value between 0 to 1, default:1.0
                      Maximum volume fraction of the inclusion family.
    a:                float, default:10
                      Diameter of the inclusion along direction-1 in actual units (mm/cm).
    b:                float, default:b=a* average_shape[1]         
                      Diameter of the inclusion along direction-2 in actual units (mm/cm).
    c:                float, default:c=a* average_shape[2]
                      Diameter of the inclusion along direction-3 in actual units (mm/cm).
    n_cuts:           int, default:10
                      Number of faces of the irregular polyhedron.
    concave:          bool, True/False, default:False
                      Boolean for concave depression on inclusion surface.
    n_concave:        int, default:0
                      Number of concave depressions on the inclusion surface.
    depth:            float, value between 0 to 1, default:0
                      Parameter which determines depth of the concave depression from the inclusion surface.
    width:            float, default:0
                      Parameter which determines width of the concave depression on the inclusion surface.
    coat:             bool, True/False, default:False
                      Boolean for coat on inclusion. 
    t_coat:           float, default:0
                      Thickness of the coating in actual units (mm/cm).
    space:            bool, True/False, default:False
                      Boolean for space which determines gap between inclusions in micro/mesostructure.
    t_space:          flaot, default:0
                      Thickness of the spacing in actual units (mm/cm).
                      Voxel value for the coat.
    kwargs:           Other parameters, default:None
    
    '''
    def __init__(self, average_shape=False, Id=None, inclusion_list=[],
                 vf_max=1, a=10, b=0, c=0, n_cuts=10, concave=False, n_concave=0,
                 depth=0, width=0, coat=False,
                 t_coat=0, space=False, t_space=0, x=0, y=0, z=0,
                 kwargs=None):
        self.inclusion_list = inclusion_list
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
        self.vf_max = vf_max
        self.standard_inclusion = None
        self.Id = None
        self.vf = 0
        self.x = x
        self.y = y
        self.z = z

        self.__dict__.update(kwargs)
        if average_shape == False:
            average_shape=np.array([1,1,1])
        self.resolution = np.array([1, 1, 1])
        self.average_shape = np.array(average_shape).astype(float)
        
        self.a = float(self.a)
        self.b = float(self.b)
        self.c = float(self.c)
        
        if self.b == 0:
            self.b = average_shape[1]*self.a
        if self.c == 0:
            self.c = average_shape[2]*self.a
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
        self.vf_max = float(self.vf_max)
        self.vox_inc = int(1)
        self.vox_coat = int(2)
        self.vox_space = int(3)

    def generate_inclusion(self):
        '''
        This method generates an inclusion

        :return: Object of class :py:class:`smg_inclusion.Inclusion`.
        '''
        inclusion = Polyhedron(a=self.a/self.resolution[0], b=self.b/self.resolution[1], c=self.c/self.resolution[2], coat=self.coat,
                                        t_coat=self.t_coat/self.resolution[0], space=self.space,
                                           t_space=self.t_space/self.resolution[0], n_cuts=self.n_cuts,
                                           concave=self.concave, n_concave=self.n_concave,
                                           depth=self.depth/self.resolution[0], width=self.width/self.resolution[0], vox_inc=self.vox_inc,
                                           vox_coat=self.vox_coat, vox_space=self.vox_space)
            
        inclusion.generate_inclusion_matrix()
        inclusion.compute_vox_volume()
        return inclusion
    
    def set_resolution(self, resolution):
        self.resolution = resolution

