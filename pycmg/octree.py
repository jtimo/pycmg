import numpy as np
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
    def __init__(self, boundary, max_points=1, depth=0):
        """ Initialize a node of the octree.
            @param boundary A Boundary object defining the cube region where points are placed
            @param max_point The maximum number of points a node can hold.
            @param depth Keeps track of how deep into the octree this node lies
            @param threshold Defines the maximum level of the octree
        """
        self.boundary = boundary
        self.max_points = max_points
        self.depth = depth
        self.points = []
        # A flag to inidcate whether this node has been divided or not.
        self.divided = False
        # A flag to indicate whether the quadrant represented by this node is full of not
        self.full = False
        self.lft = self.rft = self.lfb = self.rfb = None
        self.lbt = self.rbt = self.lbb = self.rbb = None
        
    def divide(self):
        """Branch this node by spawning eight children notdes."""
        # The naming convention of the eight children nodes follows:
        # left/right, front/back, bottm/top
        cx, cy, cz = self.boundary.centre[0], self.boundary.centre[1],self.boundary.centre[2]
        size = self.boundary.size / 2
        l, w, h = size[0], size[1], size[2]
        centre_lft = np.array([cx-w/2, cy-l/2, cz+h/2])
        centre_rft = np.array([cx+w/2, cy-l/2, cz+h/2])
        centre_lfb = np.array([cx-w/2, cy-l/2, cz-h/2])
        centre_rfb = np.array([cx+w/2, cy-l/2, cz-h/2])
        centre_lbt = np.array([cx-w/2, cy+l/2, cz+h/2])
        centre_rbt = np.array([cx+w/2, cy+l/2, cz+h/2])
        centre_lbb = np.array([cx-w/2, cy+l/2, cz-h/2])
        centre_rbb = np.array([cx+w/2, cy+l/2, cz-h/2])
        self.lft = Octree(Boundary(size,centre_lft), self.max_points, self.depth+1)
        self.rft = Octree(Boundary(size,centre_rft), self.max_points, self.depth+1)
        self.lfb = Octree(Boundary(size,centre_lfb), self.max_points, self.depth+1)
        self.rfb = Octree(Boundary(size,centre_rfb), self.max_points, self.depth+1)
        self.lbt = Octree(Boundary(size,centre_lbt), self.max_points, self.depth+1)
        self.rbt = Octree(Boundary(size,centre_rbt), self.max_points, self.depth+1)
        self.lbb = Octree(Boundary(size,centre_lbb), self.max_points, self.depth+1)
        self.rbb = Octree(Boundary(size,centre_rbb), self.max_points, self.depth+1)
        
        # Move points from parent node to children nodes
        for point in self.points:
            if self.lft.boundary.contains(point):
                self.lft.points.append(point)
            if self.rft.boundary.contains(point):
                self.rft.points.append(point)
            if self.lfb.boundary.contains(point):
                self.lfb.points.append(point)
            if self.rfb.boundary.contains(point):
                self.rfb.points.append(point)
            if self.lbt.boundary.contains(point):
                self.lbt.points.append(point)
            if self.rbt.boundary.contains(point):
                self.rbt.points.append(point)
            if self.lbb.boundary.contains(point):
                self.lbb.points.append(point)
            if self.rbb.boundary.contains(point):
                self.rbb.points.append(point)
        
        self.divided = True
        
    def insert(self, point):
        """ Try to insert point into the Octree """
        if not self.boundary.contains(point):
            # The point does not lie inside boundary: bail.
            return False
        if len(self.points) < self.max_points:
            # There's room for our point without dividing the QuadTree.
            self.points.append(point)
            return True

        # No room: divide if necessary, then try the sub-quads.
        if not self.divided:
            self.divide()
            

        return (self.lft.insert(point) or
                self.rft.insert(point) or
                self.lfb.insert(point) or
                self.rfb.insert(point) or
                self.lbt.insert(point) or
                self.rbt.insert(point) or
                self.lbb.insert(point) or
                self.rbb.insert(point))
    
    
    def __len__(self):
        ''' Return the number of points in the Octree.'''
        npoints = len(self.points)
        if self.divided:
            npoints += len(self.lft)+len(self.rft)+len(self.lfb)+len(self.rfb)+len(self.lbt)+len(self.rbt)+len(self.lbb)+len(self.rbb)-len(self.points)           
        return npoints


    def draw(self, ax):
        ''' Draw a representation of the Octree.'''
        self.boundary.draw(ax)
        if self.divided:
            self.lft.draw(ax)
            self.rft.draw(ax)
            self.lfb.draw(ax)
            self.rfb.draw(ax)
            self.lbt.draw(ax)
            self.rbt.draw(ax)
            self.lbb.draw(ax)
            self.rbb.draw(ax)
        
        
        
        