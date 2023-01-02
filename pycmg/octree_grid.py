import numpy as np
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Boundary:
    """ Boundary represents a geometric region.
        This geometric region has a shape reprensented by a size and a location."""
    def __init__(self, size, centre):
        self.size = size
        self.centre = centre
        self.left_edge, self.right_edge = centre[0] - size[0]/2, centre[0] + size[0]/2
        self.front_edge, self.back_edge = centre[1] - size[1]/2, centre[1] + size[1]/2
        self.bottom_edge, self.top_edge = centre[2] - size[2]/2, centre[2] + size[2]/2

        
    def get_size(self):
        return self.size
        
    def get_centre(self):
       return self.centre
   
    def contains(self, point):
        """ Check whether the point is inside this cube.
            @param point array of size 3, type int
            @return a boolean
        """
        return (point[0] >= self.left_edge and point[0] < self.right_edge and
                point[1] >= self.front_edge and point[1] < self.back_edge and
                point[2] >= self.bottom_edge and point[2] < self.top_edge)
    
    def draw(self, ax, c='k', lw=0.3, **kwargs):
        
        # Calculate the vertices of the cube
        v_lft = np.array([self.left_edge, self.front_edge, self.top_edge])
        v_rft = np.array([self.right_edge, self.front_edge, self.top_edge])
        v_lfb = np.array([self.left_edge, self.front_edge, self.bottom_edge])
        v_rfb = np.array([self.right_edge, self.front_edge, self.bottom_edge])
        v_lbt = np.array([self.left_edge, self.back_edge, self.top_edge])
        v_rbt = np.array([self.right_edge, self.back_edge, self.top_edge])
        v_lbb = np.array([self.left_edge, self.back_edge, self.bottom_edge])
        v_rbb = np.array([self.right_edge, self.back_edge, self.bottom_edge])
        
        # plotting cube
        # Initialize a list of vertex coordinates for each face
        # faces = [np.zeros([5,3])]*3
        faces = []
        faces.append(np.zeros([5,3]))
        faces.append(np.zeros([5,3]))
        faces.append(np.zeros([5,3]))
        faces.append(np.zeros([5,3]))
        faces.append(np.zeros([5,3]))
        faces.append(np.zeros([5,3]))
        # Bottom face
        faces[0][:,0] = [v_lfb[0],v_rfb[0],v_rbb[0],v_lbb[0],v_lfb[0]]
        faces[0][:,1] = [v_lfb[1],v_rfb[1],v_rbb[1],v_lbb[1],v_lfb[1]]
        faces[0][:,2] = [v_lfb[2],v_rfb[2],v_rbb[2],v_lbb[2],v_lfb[2]]
        # Top face
        faces[1][:,0] = [v_lft[0],v_rft[0],v_rbt[0],v_lbt[0],v_lft[0]]
        faces[1][:,1] = [v_lft[1],v_rft[1],v_rbt[1],v_lbt[1],v_lft[1]]
        faces[1][:,2] = [v_lft[2],v_rft[2],v_rbt[2],v_lbt[2],v_lft[2]]
        # Left Face
        faces[2][:,0] = [v_lfb[0],v_lft[0],v_lbt[0],v_lbb[0],v_lfb[0]]
        faces[2][:,1] = [v_lfb[1],v_lft[1],v_lbt[1],v_lbb[1],v_lfb[1]]
        faces[2][:,2] = [v_lfb[2],v_lft[2],v_lbt[2],v_lbb[2],v_lfb[2]]
        # Right Face
        faces[3][:,0] = [v_rfb[0],v_rft[0],v_rbt[0],v_rbb[0],v_rfb[0]]
        faces[3][:,1] = [v_rfb[1],v_rft[1],v_rbt[1],v_rbb[1],v_rfb[1]]
        faces[3][:,2] = [v_rfb[2],v_rft[2],v_rbt[2],v_rbb[2],v_rfb[2]]
        # Front face
        faces[4][:,0] = [v_lfb[0],v_rfb[0],v_rft[0],v_lft[0],v_lfb[0]]
        faces[4][:,1] = [v_lfb[1],v_rfb[1],v_rft[1],v_lft[1],v_lfb[1]]
        faces[4][:,2] = [v_lfb[2],v_rfb[2],v_rft[2],v_lft[2],v_lfb[2]]
        # Back face
        faces[5][:,0] = [v_lbb[0],v_rbb[0],v_rbt[0],v_lbt[0],v_lbb[0]]
        faces[5][:,1] = [v_lbb[1],v_rbb[1],v_rbt[1],v_lbt[1],v_lbb[1]]
        faces[5][:,2] = [v_lbb[2],v_rbb[2],v_rbt[2],v_lbt[2],v_lbb[2]]
        ax.add_collection3d(Poly3DCollection(faces, facecolors='white',lw=lw, edgecolors='k', alpha=0))
        

class Octree:
    """A class implementing an octree."""
    def __init__(self, size, max_points=1):
        self.size = size
        self.max_points = max_points
        #self.boundary = boundary
        self.tree = {}
    
    def level(self):
        ''' Calculate how many levels are needed to hold all the points'''
        points = np.ceil(self.size / self.max_points)
        level = np.ceil(math.log((points*7+1),8)).astype(int)
        return level
    
    def encode(self):
        level = self.level()
        for l in range(level):
            if l == 0:
                self.tree[(0,0,0,0)] = []
            else:
                for code in self.tree.copy():
                    self.encode_next_level(code)
            
    def encode_next_level(self,code):
        l,x,y,z = code[0]+1, code[1]*2, code[2]*2, code[3]*2
        self.tree[(l,x,y,z)] = []
        self.tree[(l,x+1,y,z)] = []
        self.tree[(l,x+1,y+1,z)] = []
        self.tree[(l,x,y+1,z)] = []
        self.tree[(l,x,y,z+1)] = []
        self.tree[(l,x+1,y,z+1)] = []
        self.tree[(l,x+1,y+1,z+1)] = []
        self.tree[(l,x,y+1,z+1)] = []
            
    def insert(self, point, code):
        points = self.tree[code]
        points.append(point)
        self.tree[code] = points
        
    def get_boundary(self, root, code):
        level = code[0]
        size = root.size / 2**level
        x = code[1]*size[0] + size[0]/2
        y = code[2]*size[1] + size[1]/2
        z = code[3]*size[2] + size[2]/2
        centre = np.array([x,y,z])
        boundary = Boundary(size, centre)
        return boundary
        
    def get_next_insertion(self, pre_code):
        '''Get the code for the next insertion'''
        current_level = pre_code[0]
        for x in range(2**current_level):
            for y in range(2**current_level):
                for z in range(2**current_level):
                    code = (current_level,x,y,z)
                    if not self.tree[code]:
                        return code,current_level
                    if len(self.tree[code]) < self.max_points:
                        return code,current_level
        
        current_level += 1
        return (current_level,0,0,0),current_level
    
    def initiate(self):
        self.encode()
    
    
            
    
            
            
        
        
        
