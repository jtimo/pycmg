# -*- coding: utf-8 -*-

import numpy as np
from octree_grid import Octree,Boundary

class Mesostructure:
    ''' This class generates micro/mesostructure by assembling the inclusion/aggregates on to the main micro/mesostructure

    Parameters
    ----------
    mesostructure_size:   array of size (3), type int, default:[100,100,100]
                          Size of the mesostructure 3D matrix.
    configuration:        Configuration object
                          Configuration object which provides details about the aggregate type and size distribution for assembly.
    resolution            array of size (3), type float, default: [1,1,1]
                          resolution of the mesostructure (resolution for the voxel format)
                          
    '''
    def __init__(self,mesostructure_size=[100,100,100], configuration=None, resolution=False):
        if resolution==False:
            resolution = np.array([1,1,1])
        
        self.meso_size = mesostructure_size
        self.resolution=np.array(resolution).astype(float)
        self.size = np.array(np.array(mesostructure_size).astype(float)/self.resolution).astype(int)
        self.configuration = []
        self.vf_max = []; self.vf = []; self.attempt = []
        config = []
        config.append(configuration)
        if configuration is not None:
            self.configuration.extend(config)
            
        for i in range(len(self.configuration)):
            self.vf_max.append(self.configuration[i].vf_max_assembly)
        if np.sum(self.size != 0) != 3:
            raise Exception('Assembly size is invalid')
        self.inclusion_list = []
        self.conf_count = 0
        self.n_inc_total = []
        self.mat_meso = np.zeros((self.size)).astype(int)
        self.vf_previous = 0
            
    
    def add_configuration(self, configuration):
        '''
        Add inclusion configuration to the assembly.
        
        Parameters
        ----------
        configuration:   Configuration object
                         Configuration object which provides details about the aggregate type and size distribution for assembly.
                         
        '''
        config = []
        config.append(configuration)
        self.configuration.extend(config)
        for i in range(len(config)):
            self.vf_max.append(config[i].vf_max_assembly)
    
    def assemble_sra(self, attempt_max=500000, threshold=50, iter_limit=10):
        '''
        Assemble aggregates/pores onto the mesostructure 3D matrix using Semi-Random Assembly (SRA) algorithm.

        Parameters
        ----------
        attempt_max:    int, default:500000
                        Maximum number of unsuccessfull assembly attempts before temrinating the assembly algorithm.
        threshold:      int, default:50
                        Number of unsuccessfull attempts after which the algorithm shifts to SRA (alorithm type-2) from RSA (algorithm type-1)
        iter_limit:     int, default:10
                        Number of unsuccessfull attempts to try with the same particle/aggregate orientation before switching to another random orientation.
                        
        Return
        ------
        mat_meso:       3D array of type int
                        Mesostructure 3D array with aggregates/pores/particles assembled inside.
        '''
        
        if len(self.configuration) == 0:
            raise Exception('No configuration is loaded')
            
        if len(self.configuration[self.conf_count].inclusion_fam_list) == 0:
            raise Exception('No inputs are given for the configuration. You can provide default inputs by using load_inclusion() method in Configuration class!')

        if np.sum(self.vf_max) > 1:
            raise Exception('Maximum volume fraction of the aggregates in the micro/mesostructure cannot be more than 1')
        
        
        assembly_vf_vox = np.size(self.mat_meso)
        inclusion_count = []
        vf_inc_max = 0
        
        for i in range(np.size(self.configuration[self.conf_count].inclusion_fam_list)):
            self.configuration[self.conf_count].inclusion_fam_list[i].set_resolution(self.resolution)
            vol_vox = 0
            shuffle_number = 10
            for j in range(shuffle_number):
                standard_inclusion = self.configuration[self.conf_count].inclusion_fam_list[i].generate_inclusion()
                vol_vox += standard_inclusion.vol_vox
                
            average_vol_vox = float(vol_vox)/float(shuffle_number)
            self.configuration[self.conf_count].inclusion_fam_list[i].vol_vox = average_vol_vox
            vf_inc_max += self.configuration[self.conf_count].inclusion_fam_list[i].vf_max
            self.configuration[self.conf_count].inclusion_fam_list[i].vf_each = float(self.configuration[self.conf_count].inclusion_fam_list[i].vol_vox)/float(assembly_vf_vox)
            self.configuration[self.conf_count].inclusion_fam_list[i].n_inclusion = int(np.ceil(float(self.configuration[self.conf_count].inclusion_fam_list[i].vf_max)*self.vf_max[self.conf_count]/self.configuration[self.conf_count].inclusion_fam_list[i].vf_each))
            inclusion_count.append(self.configuration[self.conf_count].inclusion_fam_list[i].n_inclusion)
            self.configuration[self.conf_count].inclusion_fam_list[i].count = 0
        
        if max(self.configuration[self.conf_count].inclusion_size_list) > np.min(self.meso_size):
            raise Exception('Inclusion size is larger than the mesostructure size')
        vf_test=np.zeros((np.size(self.configuration[self.conf_count].inclusion_fam_list)))
        vf_ttest=0
        sort = np.array(self.configuration[self.conf_count].inclusion_size_list).argsort()
        sorted_id = np.array(self.configuration[self.conf_count].inclusion_fam_id_list)[sort[::-1]]
        for i in range(np.size(self.configuration[self.conf_count].inclusion_fam_list)):
            vf_ttest+=self.configuration[self.conf_count].inclusion_fam_list[sorted_id[i]].n_inclusion*self.configuration[self.conf_count].inclusion_fam_list[sorted_id[i]].vf_each 
            vf_test[i]=vf_ttest
        if vf_inc_max <= 1-10E-3 or vf_inc_max >= 1+10E-3:
            raise Exception('Total maximum volume fraction of all inclusion families must be close to 1')
        
        
        self.n_inc_total.append(np.sum(inclusion_count))
        inclusion_fam_list = self.configuration[self.conf_count].inclusion_fam_list
        
        algType = 1
        vf = 0; i = 0; attempt = 0; T = threshold
        vf_max = np.sum(self.vf_max[0:self.conf_count+1])
        print('vf_max:',vf_max)
        inclusion_list = []
        
        while vf < vf_max and i < np.size(inclusion_fam_list) and attempt <= attempt_max:
            inclusion = inclusion_fam_list[sorted_id[i]].generate_inclusion()
            iteration = 0
            # SWitch to semi-random assembly if attempts exceed threshold
            if attempt > T:
                algType = 2
            else:
                algType = 1
            
            # Swich to another aggregate if iterations exceed iteration limit
            accept = 0
            while accept == 0 and iteration < iter_limit and attempt <= attempt_max:
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
                        while accept == 0 and iteration <= iter_limit and attempt <= attempt_max and algType == 2:
                            iteration = iteration+1
                            x0 = np.round(np.random.random(3)*self.size-1).astype(int)
                            direction = np.random.permutation(3)
                            while accept == 0 and x0[direction[1]] < self.size[direction[1]] and attempt <= attempt_max:
                                while accept == 0 and x0[direction[0]] < self.size[direction[0]] and attempt <= attempt_max:
                                    if self.mat_meso[x0[0],x0[1],x0[2]] == 0:
                                        self.mat_meso,inclusion,check = self.__assemble_inclusion(self.mat_meso, inclusion, x0)
                                        if check is True:
                                            accept = 1
                                            inclusion.x0 = np.copy(x0)
                                            
                                    x0[direction[0]] += 1
                                    attempt += 1
                                x0[direction[0]] = 0
                                x0[direction[1]] += 1
                else:
                    attempt += 1

            if accept == 1:
                inclusion_list.append(inclusion)
                inclusion_fam_list[sorted_id[i]].inclusion_list.append(inclusion)
                inclusion_fam_list[sorted_id[i]].count += 1
                inclusion.vol_vox = np.sum(inclusion.mat_inc == inclusion.vox_inc)+np.sum(inclusion.mat_inc==inclusion.vox_coat)
                inclusion.vf_each = inclusion.vol_vox/np.size(self.mat_meso)
                inclusion_fam_list[i].vf += inclusion.vf_each
                vf = np.sum(np.logical_and(self.mat_meso != 0, self.mat_meso != inclusion_fam_list[i].vox_space)) / np.size(self.mat_meso)
                T=threshold
                iteration = 0
                accept = 0
                attempt = 0
                if inclusion_fam_list[sorted_id[i]].count >= inclusion_fam_list[sorted_id[i]].n_inclusion:
                    print('i:', i)
                    print('count:', inclusion_fam_list[sorted_id[i]].count)
                    i += 1 
        self.vf.append(vf); self.attempt.append(attempt)
        self.inclusion_list.append(inclusion_list)
        print('size of inclusion list:', len(inclusion_list))
        self.conf_count += 1
        print('Configuration {0} is assembled with volume fraction {1}'.format(self.conf_count,vf-self.vf_previous))
        self.vf_previous += vf
        return self.mat_meso


    def assemble_octree(self, attempt_max=500000, threshold=20, iter_limit=10,volfrac_limit = 0.6):
        '''
        Assemble aggregates/pores onto the mesostructure 3D matrix using Semi-Random Assembly (SRA) algorithm.

        Parameters
        ----------
        attempt_max:    int, default:500000
                        Maximum number of unsuccessfull assembly attempts before temrinating the assembly algorithm.
        threshold:      int, default:50
                        Number of unsuccessfull attempts after which the area shifts to next boundary
        iter_limit:     int, default:10
                        Number of unsuccessfull attempts to try with the same particle/aggregate orientation before switching to another random orientation.
                        
        Return
        ------
        mat_meso:       3D array of type int
                        Mesostructure 3D array with aggregates/pores/particles assembled inside.
        '''
        if len(self.configuration) == 0:
            raise Exception('No configuration is loaded')            
        if len(self.configuration[self.conf_count].inclusion_fam_list) == 0:
            raise Exception('No inputs are given for the configuration. You can provide default inputs by using load_inclusion() method in Configuration class!')
        if np.sum(self.vf_max) > 1:
            raise Exception('Maximum volume fraction of the aggregates in the micro/mesostructure cannot be more than 1')
        
        assembly_vf_vox = np.size(self.mat_meso)
        inclusion_count = []
        vf_inc_max = 0
        
        for i in range(np.size(self.configuration[self.conf_count].inclusion_fam_list)):
            self.configuration[self.conf_count].inclusion_fam_list[i].set_resolution(self.resolution)
            vol_vox = 0
            shuffle_number = 10
            for j in range(shuffle_number):
                standard_inclusion = self.configuration[self.conf_count].inclusion_fam_list[i].generate_inclusion()
                vol_vox += standard_inclusion.vol_vox
                
            average_vol_vox = float(vol_vox)/float(shuffle_number)
            self.configuration[self.conf_count].inclusion_fam_list[i].vol_vox = average_vol_vox
            vf_inc_max += self.configuration[self.conf_count].inclusion_fam_list[i].vf_max
            self.configuration[self.conf_count].inclusion_fam_list[i].vf_each = float(self.configuration[self.conf_count].inclusion_fam_list[i].vol_vox)/float(assembly_vf_vox)
            self.configuration[self.conf_count].inclusion_fam_list[i].n_inclusion = int(np.ceil(float(self.configuration[self.conf_count].inclusion_fam_list[i].vf_max)*self.vf_max[self.conf_count]/self.configuration[self.conf_count].inclusion_fam_list[i].vf_each))
            inclusion_count.append(self.configuration[self.conf_count].inclusion_fam_list[i].n_inclusion)
            self.configuration[self.conf_count].inclusion_fam_list[i].count = 0
        
        if max(self.configuration[self.conf_count].inclusion_size_list) > np.min(self.meso_size):
            raise Exception('Inclusion size is larger than the mesostructure size')
        vf_test=np.zeros((np.size(self.configuration[self.conf_count].inclusion_fam_list)))
        vf_ttest=0
        sort = np.array(self.configuration[self.conf_count].inclusion_size_list).argsort()
        sorted_id = np.array(self.configuration[self.conf_count].inclusion_fam_id_list)[sort[::-1]]
        for i in range(np.size(self.configuration[self.conf_count].inclusion_fam_list)):
            vf_ttest+=self.configuration[self.conf_count].inclusion_fam_list[sorted_id[i]].n_inclusion*self.configuration[self.conf_count].inclusion_fam_list[sorted_id[i]].vf_each 
            vf_test[i]=vf_ttest
        if vf_inc_max <= 1-10E-3 or vf_inc_max >= 1+10E-3:
            raise Exception('Total maximum volume fraction of all inclusion families must be close to 1')
        
        
        self.n_inc_total.append(np.sum(inclusion_count))
        inclusion_fam_list = self.configuration[self.conf_count].inclusion_fam_list
        
        # Initialization for assembling
        vf = 0; i = 0; attempt = 0; T = threshold
        vf_max = np.sum(self.vf_max[0:self.conf_count+1])
        inclusion_list = []        
        
        # Initialize Octree
        tree_size = self.n_inc_total[0]
        max_points = 1
        tree = Octree(tree_size,max_points)
        tree.initiate()
        root = Boundary(self.size, self.size/2)
        init = 0
        current_level = 0
        code = (0,0,0,0)
        while vf < vf_max and i < np.size(inclusion_fam_list) and attempt <= attempt_max:
            inclusion = inclusion_fam_list[sorted_id[i]].generate_inclusion()
            iteration = 0
            accept = 0
            
            while accept == 0 and iteration < iter_limit and attempt <= attempt_max:
                iteration = iteration+1
                
                if attempt > T:
                    x0 = np.floor(np.random.random(3)*(self.size-1)).astype(int)
                else:
                    if init == 0:
                        boundary = root
                        length = np.ravel(np.multiply(np.random.uniform(-1,1,(1,3)), (boundary.size-1)/2)).astype(int)
                        x0 = boundary.centre.astype(int) + length
                        init = 1
                    else:
                        code,current_level = tree.get_next_insertion(code)
                        boundary = tree.get_boundary(root, code)
                        length = np.ravel(np.multiply(np.random.uniform(-1,1,(1,3)), (boundary.size-1)/2)).astype(int)
                        x0 = boundary.centre.astype(int) + length
                        c,l = (boundary.centre).astype(int),((boundary.size-1)/2).astype(int)
                        snippet = self.mat_meso[c[0]-l[0]:c[0]+l[0],c[1]-l[1]:c[1]+l[1],c[2]-l[2]:c[2]+l[2]]
                        volfrac_s = np.count_nonzero(snippet == 1) / (snippet.shape[0]*snippet.shape[1]*snippet.shape[2])
                        if volfrac_s > volfrac_limit:
                            tree.insert(np.array([0,0,0]), code)
                            attempt = 0
                            continue
                
                #x0 = np.floor(np.random.random(3)*(self.size-1)).astype(int)
                if self.mat_meso[x0[0],x0[1],x0[2]] == 0:
                    self.mat_meso,inclusion,check = self.__assemble_inclusion(self.mat_meso, inclusion, x0)
                    if check == True:
                        accept = 1
                        tree.insert(np.array([1,1,1]), code)
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
                inclusion.vol_vox = np.sum(inclusion.mat_inc == inclusion.vox_inc)+np.sum(inclusion.mat_inc==inclusion.vox_coat)
                inclusion.vf_each = inclusion.vol_vox/np.size(self.mat_meso)
                inclusion_fam_list[i].vf += inclusion.vf_each
                vf = np.sum(np.logical_and(self.mat_meso != 0, self.mat_meso != inclusion_fam_list[i].vox_space)) / np.size(self.mat_meso)
                T=threshold
                iteration = 0
                accept = 0
                attempt = 0
                if inclusion_fam_list[sorted_id[i]].count >= inclusion_fam_list[sorted_id[i]].n_inclusion:
                    print('i:', i)
                    print('count:', inclusion_fam_list[sorted_id[i]].count)
                    i += 1 
        self.vf.append(vf); self.attempt.append(attempt)
        self.inclusion_list.append(inclusion_list)
        #print('size of inclusion list:', len(inclusion_list))
        self.conf_count += 1
        print('Configuration {0} is assembled with volume fraction {1}'.format(self.conf_count,vf-self.vf_previous))
        self.vf_previous += vf
        #for key,value in tree.tree.items():
           # print('code:',key)
            #print('points:',value)
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

        
       
        

                        