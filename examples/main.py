# -*- coding: utf-8 -*-
import sys,os
sys.path.append('./../pycmg/')
from generate_mesostructure import Mesostructure
from configuration import Configuration
from visualization import export_data, visualize_sections

my_configuration = Configuration(vf_max_assembly=0.1)
my_configuration.load_inclusions(conf_csv='AB8_CMG.csv')
my_configuration.sort_inclusions()
my_mesostructure = Mesostructure(mesostructure_size=[50, 50, 50])
my_mesostructure.add_configuration(my_configuration)
my_synthetic_microstructure = my_mesostructure.assemble_SRA()
visualize_sections(my_synthetic_microstructure, 2)
export_data(my_synthetic_microstructure, 'vtk', 'mesostructure.vti')
