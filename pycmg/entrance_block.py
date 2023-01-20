import numpy as np

class Entrance_block():
    
    def __init__(self, mat_meso, inclusion, ref_p):
        self.mat_meso = mat_meso
        self.inclusion = inclusion
        self.ref_p = ref_p
        
    def calculate_entrance_block(self):
        indices_meso = list(zip(*np.where(self.mat_meso == 1)))
        indices_inc = np.array(list(zip(*np.where(self.inclusion == 1)))).astype(int)
        
        for i in indices_meso:
            b = np.array(i).astype(int)
            entrance_point_x = b[0] - indices_inc[:,0] + self.ref_p[0]
            entrance_point_y = b[1] - indices_inc[:,1] + self.ref_p[1]
            entrance_point_z = b[2] - indices_inc[:,2] + self.ref_p[2]
            
            self.mat_meso[entrance_point_x,entrance_point_y,entrance_point_z] = -1
        
        return self.mat_meso
        
    