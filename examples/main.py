# -*- coding: utf-8 -*-
import sys,os
sys.path.append('./../pycmg/')
from generate_mesostructure import Mesostructure
from configuration import Configuration
from visualization import export_data, visualize_sections

polyhedrons = {'inclusion_type': 'Polyhedron',
               'a': 10,
               'b': 10,
               'c': 10,
               'n_cuts': 10,
               'vf_max': 0.5,
               'vox_inc': 5,
               'coat': False,
               't_coat': 2}

ellipsoids = {'inclusion_type': 'Ellipsoid',
              'a': 5,
              'b': 5,
              'c': 5,
              'vf_max': 0.5,
              'vox_inc': 6,
              'coat': False,
              't_coat': 2}

my_configuration = Configuration(vf_max_assembly=0.1)

my_configuration.load_inclusions(conf_dict=polyhedrons)
my_configuration.load_inclusions(conf_dict=ellipsoids)

my_configuration.sort_inclusions()

my_mesostructure = Mesostructure(mesostructure_size=[100, 100, 100])
my_mesostructure.add_configuration(my_configuration)
my_synthetic_microstructure = my_mesostructure.assemble_SRA()

visualize_sections(my_synthetic_microstructure, 2)
export_data(my_synthetic_microstructure, 'vtk', 'mesostructure.vti')
