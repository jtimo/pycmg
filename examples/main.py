# -*- coding: utf-8 -*-
import sys,os
sys.path.append('./../pycmg/')
from generate_mesostructure import Mesostructure
from configuration_simp import Configuration
from visualization import export_data, visualize_sections
import numpy as np


my_configuration = Configuration(vf_max_assembly=0.3, average_shape=[1, 0.5, 0.5])
my_configuration.load_inclusions(conf_csv='AB8_CMG.csv')
my_mesostructure = Mesostructure(my_configuration, mesostructure_size=[20, 20, 20], resolution=[0.125, 0.125, 0.125])
my_virtual_mesostructure = my_mesostructure.assemble_sra()
#np.save('mesostructure.npy', my_virtual_mesostructure)    # .npy extension is added if not given
visualize_sections(my_virtual_mesostructure, 3)
#export_data(my_virtual_mesostructure, 'vtk', 'mesostructure.vti')
