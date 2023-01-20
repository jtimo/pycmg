import numpy as np
from entrance_block import Entrance_block
from visualization import visualize_sections
from inclusion import Polyhedron

def assemble_inclusion(size, mat_meso, inclusion, x0):
    check = False
    indices = lambda x_start, x_end, length: np.mod(np.arange(x_start, x_end+1), length).astype(int)
    inclusion_size = np.array(np.shape(inclusion.mat_inc))
    ind_start = x0-np.floor(inclusion_size/2)
    ind_end = x0+np.ceil(inclusion_size/2)-1
    ix = indices(ind_start[0], ind_end[0], size[0])
    iy = indices(ind_start[1], ind_end[1], size[1])
    iz = indices(ind_start[2], ind_end[2], size[2])
    [x, y, z] = np.meshgrid(ix, iy, iz)
    mat_test = mat_meso[x, y, z]
    if np.sum(mat_test[inclusion.mat_inc > 0]) == 0:
        mat_test[inclusion.mat_inc > 0] = inclusion.mat_inc[inclusion.mat_inc > 0]
        mat_meso[x, y, z] = mat_test
        check = True
    return mat_meso, check


inc_a = Polyhedron(a=5, b=10, c=11, coat=None, t_coat=None,
                 space=None, t_space=None, n_cuts=10, concave=None, 
                 n_concave=None, depth=None, width=None, vox_inc=int(1), 
                 vox_coat=int(2), vox_space=int(3))
mat_a = inc_a.generate_inclusion_matrix()

inc_b = Polyhedron(a=20, b=20, c=20, coat=None, t_coat=None,
                 space=None, t_space=None, n_cuts=10, concave=None, 
                 n_concave=None, depth=None, width=None, vox_inc=int(1), 
                 vox_coat=int(2), vox_space=int(3))
mat_b = inc_b.generate_inclusion_matrix()

size = np.array([50,50,50])
x0 = size / 2
meso = np.zeros(size).astype(int)
meso, check = assemble_inclusion(size, meso, inc_b, x0)
indices_b = np.where(meso == 1)

ref_p = np.array([2,5,5])
entrance = Entrance_block(meso, mat_a, ref_p)
meso = entrance.calculate_entrance_block()
meso[indices_b] = 2

visualize_sections(meso)
trial = np.array([[1, 1],[2, 0]])







