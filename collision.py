# This program was writen by ethan parker.
# This program contain various functions related to 3D collision detection.

from constants import *
from render_math import *
from pygame import Vector3 as Vect3


def PointOnSphere(origin=Vect3(0, 0, 0), radius=1, point=Vect3(0, 0, 1)):
    """ This function finds the nearest point on a sphere. """
    # Get unit vector facing from the sphere's origin to the point:
    sphereToPoint = point - origin
    try:
        sphereToPoint.normalize()
    except:
        pass

    # Scale vector to radius:
    return sphereToPoint * radius


def sphereIntersectTrigon(lastpos=Vect3, pos=Vect3, radius=float, face=list, edges=tuple, plane=Plane, distance=float):
    """ This function checks if a sphere is intersecting with a polygon. returns True on intersect. """
    
    radiusSquared = radius * radius
    
    # Check that an intersection is possible
    if radius > distance > -radius:

        # Check that the sphere is not on the other side of the wall (avoids getting sucked through thin walls & ledges):
        if (plane.p - lastpos).dot(plane.n) < 0:
            # Check each edge of polygon, This is catastrophically bad.
            
            if True in [((p - pos) * (p - pos)) < radiusSquared for p in edges]:
                return True

            # find point on plane and if point is within the bounds of the polygon: 
            elif pointOnTrigon(plane.vectPlaneIntersect(pos, pos + (plane.n * distance)), face): 
                return True
    return False


class MeshCollider():
    def __init__(self, mesh=Mesh, position=Vect3(0,0,0), rotation=[0,0,0], isDynamic=False):
        """ This class handles collisions with a mesh object. """
        
        # Variables:
        self.polygons = None    # <-- None of these have a type defined because any gen_ method resets the type to list.
        self.position = position
        self.rotation = rotation
        self.vertecies = None
        self.edges = None
        self.planes = None
        self.bounds = None
        self.enabled = True

        # Setup:
        self.gen_mesh(mesh)
        self.gen_edges()
        self.gen_planes()
        self.update_bounds()
        self.length = len(self.polygons)
        
    def gen_mesh(self, mesh=Mesh):
        """ This method generates the list of polygons from a Mesh object. see Mesh class in render_math.py """
        # Clear polygons
        self.polygons = []

        # Since only the points are needed for the colider, discard everything besides the vertices.
        self.polygons.extend([(face[0][0], face[1][0], face[2][0]) for face in mesh])
        
        self.vertecies = mesh.vertices

        # Convert to tuple for faster read times:
        self.polygons = tuple(self.polygons)

    def gen_edges(self):
        """ This method generates points along the edges of each polygon so that I can do the most braindead method for collision. """
        # Clear edges
        self.edges = []

        # Iterate through each face, generate evenly spaced points along its perimeter.
        for face in self.polygons:
            a, b, c = dc(face)                      # using deep copy to avoid moveing vertecies twice when .move() is called
            edge_pairs = [[a, b], [b, c], [c, a]]   # List of possible edges from the points a, b, c.
            edges = []                              # List of points along the perimeter of the polygon.
            
            for current_edge in edge_pairs:
                # Find number of steps to evenly fill gap between points:
                step = int((current_edge[0].distance_to(current_edge[1]) / MESH_COLLIDER_GAP)) 
                
                if step != 0:
                    # get inverse of step here to allow for minimal divisions.
                    inv_step = 1/step

                    # Solve for each point on the edge:
                    for x in range(step):
                        edges.append(current_edge[0].lerp(current_edge[1], (x * inv_step)))
            
            # Append interpolated edges to list of edges.
            self.edges.append(tuple(edges))

        # Convert to tuple for faster read times:
        self.edges = tuple(self.edges)

    def gen_planes(self):
        """ This method generates the list of planes from the mesh. """
        # Clear planes
        self.planes = []
        
        # Generate plane for each face in mesh:
        self.planes.extend([Plane(face[0], face[1], face[2]) for face in self.polygons])
        
        # Convert to tuple for faster read times:
        self.planes = tuple(self.planes)

    def update_bounds(self):
        """ This method calculates the min and max X,Y,Z. used to check if the player is in range. """
        minVect = Vect3(min([v.x for v in self.vertecies]), min([v.y for v in self.vertecies]), min([v.z for v in self.vertecies]))
        maxVect = Vect3(max([v.x for v in self.vertecies]), max([v.y for v in self.vertecies]), max([v.z for v in self.vertecies]))
        self.bounds = (minVect, maxVect)

    def move(self, pos):
        """ This method moves the mesh collider. """
        
        # update vertices:
        [v.update(v.x + pos.x, v.y + pos.y, v.z + pos.z) for v in self.vertecies]
        
        # update edges:
        for current_edge in self.edges:
            [v.update(v.x + pos.x, v.y + pos.y, v.z + pos.z) for v in current_edge]
        
        # update planes:
        for plane in self.planes:
            plane.p += pos

        self.bounds = (self.bounds[0] + pos, self.bounds[1] + pos)

    def rotate(self, rot):
        """ This method rotates the mesh collider. """
        
        xa, ya, za = radians(rot[0]), radians(rot[1]), radians(rot[2])
        xrm, yrm, zrm = (cos(xa),  sin(xa)), (cos(ya),  sin(ya)), (cos(za),  sin(za))

        # update planes:
        for plane in self.planes:
            plane.rotate_x(rot[0])
            plane.rotate_y(rot[1])
            plane.rotate_z(rot[2])
            
        # Update edges:
        for current_edge in self.edges:  
            [rotate_xyz_dummy(v, xrm, yrm, zrm) for v in current_edge]
        
        # Update vertices
        [rotate_xyz_dummy(v, xrm, yrm, zrm) for v in self.vertecies]

        self.update_bounds()


    def CollideMesh(self, lastpos, pos, radius):
        """ This method handles collisions with a mesh collider. """     
        
        wallCollisions = 0   # Number of wall collisions
        floorCollisions = 0  # Number of floor collisions, useful for determining if the object is airborne.

        # Check that the player is within the bounding box of the collider.     
        boundsCheck = lambda x: (self.bounds[0][x] - radius) <= lastpos[x] <= (self.bounds[1][x] + radius)
        if boundsCheck(0) and boundsCheck(1) and boundsCheck(2):

            for i in range(self.length):
                plane, face, edges = self.planes[i], self.polygons[i], self.edges[i]

                # Get distance from point to plane. this is needed for later calculations so I did it here to avoid doing it twice.
                distance = plane.pointToPlane(pos)
                if sphereIntersectTrigon(lastpos, pos, radius, face, edges, plane, distance):        
                    
                    if plane.n[1] > 0.4:  # Face is mostly horizontal, dont slide across it.
                        floorCollisions += 1
                        pos += Vect3(0, 1, 0) * (plane.n * (radius - distance)).length()
                    
                    else:                          # Face is most Vertical, slide across it.
                        pos += plane.n * (radius - distance)
                        wallCollisions += 1

        return pos, wallCollisions, floorCollisions


class Gravity:
    """ This function approximates gravity for an object. """
    def __init__(self):
        self.amplitude = -0.04
        self.offset = 1
        self.maxAccel = 12
        self.current_time = 0
        self.current_value = 0

    def set_falling(self):
        if self.current_time < 0:
            self.current_time = 0

    def set_jumping(self):
        self.current_time = -1

    def update(self, frame_delta):
        self.current_time += frame_delta * 0.4

        if self.current_time > self.maxAccel:
            self.current_time = self.maxAccel

        self.current_value = self.amplitude * (self.current_time * self.current_time - self.offset) + 2

