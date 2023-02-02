# -*- coding: utf-8 -*-

import numpy as np
from inclusion_simp import Polyhedron
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
           :widths: 1,1,1,1,1

        .. note::
          **The header of the parameters in the csv file should be as follows:**

          - a:              diameter of the inclusion along direction-1 in actual units (mm/cm etc.).
          - b:              diameter of the inclusion along direction-2 in actual units (mm/cm etc.).
          - c:              diameter of the inclusion along direction-3 in actual units (mm/cm etc.).
          - vf_max:         Maximum volume fraction of each sized aggregates (value between 0 to 1).
          - n_cuts:         Number of faces/cuts for the polyhedron shaped aggregates.


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
    kwargs:           Other parameters, default:None
    
    '''
    def __init__(self, average_shape=False, Id=None, inclusion_list=[],
                 vf_max=1, a=10, b=0, c=0, n_cuts=10, kwargs=None):
        self.inclusion_list = inclusion_list
        self.vf_max = vf_max
        self.a = a
        self.b = b
        self.c = c
        self.n_cuts = n_cuts
        self.vf_max = vf_max
        self.standard_inclusion = None
        self.Id = None
        self.vf = 0

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
        self.vf_max = float(self.vf_max)
        self.vox_inc = int(1)

    def generate_inclusion(self):
        '''
        This method generates an inclusion

        :return: Object of class :py:class:`smg_inclusion.Inclusion`.
        '''
        inclusion = Polyhedron(a=self.a/self.resolution[0], b=self.b/self.resolution[1], 
                               c=self.c/self.resolution[2], n_cuts=self.n_cuts)           
        inclusion.generate_inclusion_matrix()
        inclusion.compute_vox_volume()
        return inclusion
    
    def set_resolution(self, resolution):
        self.resolution = resolution

