# -*- coding: utf-8 -*-

import numpy as np

class Mesostructure:
    ''' This class generates micro/mesostructure by assembling the inclusion/aggregates on to the main micro/mesostructure

    Parameters
    ----------
    configuration:        Configuration object
                          Configuration object which provides details about the aggregate type and size distribution for assembly.                      
    '''
    
    def __init__(self,configuration): 
        self.configuration = configuration
        self.meso_size = self.configuration.meso_size
        self.resolution = self.configuration.resolution
        self.size = self.configuration.size       
            
        if self.configuration is None:
            raise Exception('No configuration is loaded')
        
        self.vf_max = self.configuration.vf_max_assembly  
        self.mat_meso = np.zeros((self.size)).astype(int)
    
    
    def assemble_sra(self, attempt_max=500000, iter_limit=10):
        '''
        Assemble aggregates/pores onto the mesostructure 3D matrix using Semi-Random Assembly (SRA) algorithm.

        Parameters
        ----------
        attempt_max:    int, default:500000
                        Maximum number of unsuccessfull assembly attempts before temrinating the assembly algorithm.
        iter_limit:     int, default:10
                        Number of unsuccessfull attempts to try with the same particle/aggregate orientation before switching to another random orientation.                       
       
        Return
        ------
        mat_meso:       3D array of type int
                        Mesostructure 3D array with aggregates/pores/particles assembled inside.
        '''
        
        inclusion_fam_list = self.configuration.inclusion_fam_list
        sorted_id = self.configuration.sorted_id
        
        vf = 0; i = 0; attempt = 0
        vf_max = self.vf_max
        print('vf_max:',vf_max)
        inclusion_list = []
        
        while vf < vf_max and i < np.size(inclusion_fam_list) and attempt <= attempt_max:
            inclusion = inclusion_fam_list[sorted_id[i]].generate_inclusion()
            iteration = 0
            accept = 0            
            # Swich to another aggregate if iterations exceed iteration limit
            while accept == 0 and iteration < iter_limit:
                iteration = iteration+1
                x0 = np.floor(np.random.random(3)*(self.size-1)).astype(int)
                if self.mat_meso[x0[0],x0[1],x0[2]] == 0:
                    self.mat_meso,inclusion,check = self.__assemble_inclusion(self.mat_meso, inclusion, x0)
                    if check == True:
                        accept = 1
                        inclusion.x0 = np.copy(x0)
                    else:
                        accept = 0
                        attempt = attempt+1           
                else:
                    attempt += 1

            if accept == 1:
                inclusion_list.append(inclusion)
                inclusion_fam_list[sorted_id[i]].inclusion_list.append(inclusion)
                inclusion_fam_list[sorted_id[i]].count += 1
                inclusion.vol_vox = np.sum(inclusion.mat_inc == inclusion.vox_inc)
                inclusion.vol_vox = np.sum(inclusion.mat_inc == inclusion.vox_inc)
                inclusion.vf_each = inclusion.vol_vox/np.size(self.mat_meso)
                inclusion_fam_list[i].vf += inclusion.vf_each
                vf = np.sum(self.mat_meso != 0) / np.size(self.mat_meso)
                iteration = 0
                accept = 0
                attempt = 0
                
                if inclusion_fam_list[sorted_id[i]].count >= inclusion_fam_list[sorted_id[i]].n_inclusion:
                    i += 1 
                
        print('Configuration is assembled with volume fraction:', vf)
        return self.mat_meso


    def __assemble_inclusion(self, mat_meso, inclusion, x0):
        check = False
        indices = lambda x_start, x_end, length: np.mod(np.arange(x_start, x_end+1), length).astype(int)
        inclusion_size = np.array(np.shape(inclusion.mat_inc))
        ind_start = x0-np.floor(inclusion_size/2)
        ind_end = x0+np.ceil(inclusion_size/2)-1
        ix = indices(ind_start[0], ind_end[0], self.size[0])
        iy = indices(ind_start[1], ind_end[1], self.size[1])
        iz = indices(ind_start[2], ind_end[2], self.size[2])
        [x, y, z] = np.meshgrid(ix, iy, iz)
        mat_test = mat_meso[x, y, z]
        if np.sum(mat_test[inclusion.mat_inc > 0]) == 0:
            mat_test[inclusion.mat_inc > 0] = inclusion.mat_inc[inclusion.mat_inc > 0]
            mat_meso[x, y, z] = mat_test
            check = True
        return mat_meso, inclusion, check

        
       
        

                        