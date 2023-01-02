from octree_grid import Octree,Boundary
import numpy as np


'''Test the pipeline'''

tree_size = 30
max_points = 4
tree_p = Octree(tree_size,max_points)
tree_p.initiate()

root_p = Boundary(np.array([40,40,40]), np.array([20,20,20]))
code = (0,0,0,0)

for i in range(tree_size):
    if i == 0:
        boundary = root_p
        length = np.ravel(np.multiply(np.random.uniform(-1,1,(1,3)), boundary.size/2).astype(int))
        init = boundary.centre + length
        tree_p.insert(init, code)
        print(boundary.centre)
        print(length)
        print(init)
    else:
        code,current_level = tree_p.get_next_insertion(code)
        boundary = tree_p.get_boundary(root_p, code)
        point = boundary.centre + np.multiply(np.random.uniform(-1,1,(1,3)), boundary.size/2).astype(int)
        tree_p.insert(point, code)


for key,value in tree_p.tree.items():
    print('code:',key)
    print('points:',value)


