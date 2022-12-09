# This program was writen by ethan parker.
# This program contain various functions related to 3D collision detection.

from constants import *
from render_math import *
from pygame import Vector3 as Vect3


def PointOnSphere(origin=Vect3(0, 0, 0), radius=1, point=Vect3(0, 0, 1)):
    """ This function finds the nearest point on a plane. """
    # Get unit vector facing from the sphere's origin to the point:
    sphereToPoint = point - origin
    try:
        sphereToPoint.normalize()
    except:
        pass

    # Scale vector to radius:
    nearest = sphereToPoint * radius
    return nearest


class MeshCollider():
    def __init__(self, mesh):
        """ This class handles collisions with a mesh object. """
        
        # Variables:
        self.mesh = None
        self.planes = None
        self.enabled = True

        # Setup:
        self.gen_mesh(mesh)
        self.gen_planes()

    def gen_mesh(self, mesh):
        """ This method generates the list of polygons from a Mesh object. see Mesh class in render_math.py """
        self.mesh = []

        # Since only the points are needed for the colider, discard everything besides the vertices.
        self.mesh.extend([(face[0][0], face[1][0], face[2][0]) for face in mesh])
        self.mesh = tuple(self.mesh)

    def gen_planes(self):
        """ This method generates the list of planes from the mesh. """
        self.planes = []
        
        self.planes.extend([Plane(face[0], face[1], face[2]) for face in self.mesh])
        self.planes = tuple(self.planes)


    def CollideMesh(self, pos, vel, radius):
        """ This method handles collisions with a mesh colider. """     

        delta_shift = Vect3(0, 0, 0)  # Stores the Vect3 that unintersects the sphere from the mesh.
        collisions = 0                # Stores the number of collisions, usefull for determining if the object is airborne.
        nextpos = pos + vel           # The position the sphere will be at after this frame assuming no colisions.

        collideWithFloor = False      # Bool for if collision with floor is detected (normal is mostly vertical).

        for i in range(len(self.planes)):
            
            # Check that the sphere is not on the other side of the wall (avoids getting sucked off ledges):
            if (pos - self.planes[i].p).normalize().dot(self.planes[i].n) > 0:
                distance = self.planes[i].pointToPlane(nextpos)
                
                if radius > distance > -radius:
                    a, b, c = self.mesh[i]
                    intersect = self.planes[i].vectPlaneIntersect(nextpos, nextpos + (self.planes[i].n * distance))
                    
                    # Check that the intersection point is within the bounds of the triangle: 
                    if pointOnTrigon(intersect, a, b, c):
                        
                        # Unintersect sphere with face:
                        if self.planes[i].n[1] > 0.6:  # Face is mostly Horiziontal, dont slide across it.
                            collideWithFloor = True
                            shift = Vect3(0, 1, 0) * (self.planes[i].n * (radius - distance)).length()
                        
                        else:                          # Face is most Vertical, slide across it.
                            shift = Vect3(self.planes[i].n * (radius - distance))
                        
                        delta_shift += shift
                        collisions += 1

        return delta_shift + nextpos, collisions, collideWithFloor


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

