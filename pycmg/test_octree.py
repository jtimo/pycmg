import numpy as np
import matplotlib.pyplot as plt
from octree import Boundary, Octree

size_init = np.array([200, 200, 200])
centre_init = np.array(size_init / 2)
boundary = Boundary(size_init, centre_init)
octree = Octree(boundary, max_points=2)

# Generate random 3D points
N = 15
coords = np.random.rand(N,3)
points = coords * size_init[0]
#points = np.append(points,np.array([300, 300, 300]))
#points = [np.array([50,50,100]), np.array([150,50,100]), np.array([300, 200, 200])]

for point in points:
    print('Insert successfully:', octree.insert(point))

print('Number of points to be inserted = ', len(points))
print('Number of points in the domain =', len(octree))
print(octree.divided)

# Plot
# First initialize the fig variable to a figure
DPI = 72
fig = plt.figure(figsize=(300/DPI, 300/DPI), dpi=DPI)
# Add a 3d axis to the figure
ax = fig.add_subplot(projection='3d')
# Plot the division
octree.draw(ax)
# Plot the points
ax.scatter([p[0] for p in points], [p[1] for p in points], [p[2] for p in points],s=4)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.show()




