# This program was writen by Ethan Parker
# This program consolidates all the mathematics used in rendering.

from math import *
from constants import *
from pygame import Vector3 as Vect3
from pygame import Vector2 as Vect2
from copy import deepcopy as dc


def clamp(n, min, max):
    if n < min:
        return min
    elif n > max:
        return max
    else:
        return n


def rotate_x_dummy(vect, cos_sin):
    y, z = vect.y, vect.z
    vect.y, vect.z = (y * cos_sin[0]) - (z * cos_sin[1]), (z * cos_sin[0]) + (y * cos_sin[1])


def rotate_y_dummy(vect, cos_sin):
    x, z = vect.x, vect.z
    vect.x, vect.z = (x * cos_sin[0]) - (z * cos_sin[1]), (z * cos_sin[0]) + (x * cos_sin[1])


def rotate_z_dummy(vect, cos_sin):
    x, y = vect.x, vect.y
    vect.x, vect.y = (x * cos_sin[0]) - (y * cos_sin[1]), (y * cos_sin[0]) + (x * cos_sin[1])


def rotate_xyz_dummy(vect, xrm, yrm, zrm):
    x, y, z = vect.x, vect.y, vect.z
    y, z = (y * xrm[0]) - (z * xrm[1]), (z * xrm[0]) + (y * xrm[1])
    x, z = (x * yrm[0]) - (z * yrm[1]), (z * yrm[0]) + (x * yrm[1])
    x, y = (x * zrm[0]) - (y * zrm[1]), (y * zrm[0]) + (x * zrm[1])
    vect.x, vect.y, vect.z = x, y, z

def smoothLerp(a, b, t):
    """ Returns the smoothly interpolated value between a and b at t in range [0,1]. """
    return Lerp(a, b, Lerp(t*t, 1-((t-1)*(t-1)), t))


def Lerp(a, b, t):
    """ Returns the interpolated value between a and b at t in range [0,1]."""
    return a + (b - a) * t    


def pointOnTrigon(p, points):
    """ This function handles checking of a point "p" is within the bounds of a Trigon. """
    
    # Get the points offset by p.
    a, b, c = points[0] - p, points[1] - p, points[2] - p
    
    # Get normals of faces formed by a, b, c to p:
    u = b.cross(c)
    # if the normals are not parallel, the point does not intersect the face.
    if u.dot(c.cross(a)) > 0.0001 and u.dot(a.cross(b)) > 0.0001:
        return True

    return False


class Plane:
    """ This class defines a basic 3d plane as defined by 3 points. """
    def __init__(self, a, b, c):
        self.n = getNormal(a, b, c)
        self.p = (a + b + c) * ONE_THIRD
        self.d = self.p.dot(self.n)

    def rotate_x(self, angle):
        """ This method rotates the plane around the X axis.
            NOTE: .update() method must be called after rotation to update self.d """
        angle = radians(angle)
        cos_sin = (cos(angle), sin(angle))
        rotate_x_dummy(self.n, cos_sin)
        rotate_x_dummy(self.p, cos_sin)
    
    def rotate_y(self, angle):
        """ This method rotates the plane around the Y axis. 
            NOTE: .update() method must be called after rotation to update self.d """
        angle = radians(angle)
        cos_sin = (cos(angle), sin(angle))
        rotate_y_dummy(self.n, cos_sin)
        rotate_y_dummy(self.p, cos_sin)

    def rotate_z(self, angle):
        """ This method rotates the plane around the Z axis. 
            NOTE: .update() method must be called after rotation to update self.d """
        angle = radians(angle)
        cos_sin = (cos(angle), sin(angle))
        rotate_z_dummy(self.n, cos_sin)
        rotate_z_dummy(self.p, cos_sin)

    def move(self, position):
        """ This function moves the plane to XYZ. """
        self.p += position

    def update(self):
        self.d = self.p.dot(self.n) 

    def pointToPlane(self, p):
        """ This function calculates the point-to-plane distance from any given point. """
        return self.n.dot(p) - self.d

    def vectPlaneIntersect(self, start, end):
        """ This function calculates the intersection point of a vector and a plane. """
        ad = start.dot(self.n)
        return ((end - start) * ((self.d - ad) / ((end.dot(self.n)) - ad))) + start

    def vertexPlaneIntersect(self, start, end):
        """ This function calculates the intersection point of a vertex with texture coordinate and a plane. """
        ad = start[0].dot(self.n)
        t = (self.d - ad) / ((end[0].dot(self.n)) - ad)
        return ((end[0] - start[0]) * t) + start[0], ((end[1] - start[1]) * t) + start[1], ((end[2] - start[2]) * t) + start[2]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# MESH CLASS & METHODS


class Mesh:
    """ This class stores a 3D object consisting of vertices and polygons that connect said vertices.
        The structure of this object is identical to the .obj file format. """
    def __init__(self, polygons=[], polygonIndex=[], vertices=[], brightness=[], position=Vect3(0, 0, 0), static=False, updateLighting=True, texIndex=0):
        """ Vertices must be a list containing Vect3 [point 1, point 2, point 3 ...]
            Polygons must be a list containing lists containing soft copies of vertices forming a face: [[1,2,3], ...]. """

        self.vertices = vertices        # List of vertices.
        self.polygons = polygons        # List of faces connecting vertices.

        self.polygonI = polygonIndex    # List containing the index of a vertex in the vertices list.
        self.v_bright = brightness      # List of brightnesses for each vertex.

        self.farPoint = 0               # farthest point on the object. (Calculated with self.GetFarPoint)
        self.bounds = []                # Bounding box for mesh. 
        self.texIndex = texIndex        # Stores the texture index for this mesh.
        self.depth = []                 # List containing distance from camera for each face on a mesh.
        self.rotation = [0, 0, 0]       # Rotational offset.
        self.position = position        # Positional offset.
        self.origin = position          # Center point of the mesh.
        self.static = static            # Bool for if mesh is moveable.

        # Run initialization routines. 
        self.getCenter()
        self.getBounds()
        self.getFarPoint()

        if updateLighting:
            self.updateNormals()
            self.updateBrightness()

    def __len__(self):
        """ Returns the number of polygons in the mesh. """
        return len(self.polygons)

    def __getitem__(self, key):
        """ Return the polygon formed by vertices at index. """
        return self.polygons[key]

    def __delitem__(self, key):
        del self.polygons[key]

    def __setitem__(self, key, trigon):
        """ Trigon must be a list-like object, stored as: [[a,uv], [b,uv], [c,uv], normal, brightness]. """
        self.polygons[key] = trigon

    def append(self, a, b, c, n, br):
        """ This method handles appending a new trigon to the mesh. """
        self.polygons.append([a, b, c, n, br])

    def rotate(self, rot):
        """ This method rotates the mesh by XYZ. """
        xa, ya, za = radians(rot[0]), radians(rot[1]), radians(rot[2])
        xrm, yrm, zrm = (cos(xa),  sin(xa)), (cos(ya),  sin(ya)), (cos(za),  sin(za))
        [rotate_xyz_dummy(vect, xrm, yrm, zrm) for vect in self.vertices]

    def rotate_to(self, x, y, z):
        """ This method sets the XYZ rotation of the mesh. """
        dx, dy, dz = x - self.rotation[0], y - self.rotation[1], z - self.rotation[2]
        self.rotation = [x, y, z]
        xa, ya, za = radians(dx), radians(dy), radians(dz)
        xrm, yrm, zrm = (cos(xa),  sin(xa)), (cos(ya),  sin(ya)), (cos(za),  sin(za))
        [rotate_xyz_dummy(vect, xrm, yrm, zrm) for vect in self.vertices]

    def rotate_x(self, angle):
        """ This method updates the rotation of the mesh in the X axis. """
        if angle == 0: 
            return
        angle = radians(angle)
        r_math = (cos(angle),  sin(angle))
        [rotate_x_dummy(vertex, r_math) for vertex in self.vertices]

    def rotate_y(self, angle):
        """ This method updates the rotation of the mesh in the Y axis. """
        if angle == 0: 
            return
        angle = radians(angle)
        r_math = (cos(angle),  sin(angle))
        [rotate_y_dummy(vertex, r_math) for vertex in self.vertices]

    def rotate_z(self, angle):
        """ This method updates the rotation of the mesh in the Z axis. """
        if angle == 0: 
            return
        angle = radians(angle)
        r_math = (cos(angle),  sin(angle))
        [rotate_z_dummy(vertex, r_math) for vertex in self.vertices]

    def move_to(self, pos):
        """ This method moves a mesh to a specific position. """
        if self.position != pos:
            # Move to the target - the current position. (same as move to 0, then to target)
            self.position = pos
            x, y, z = pos
            [v.update(v.x + x, v.y + y, v.z + z) for v in self.vertices]

    def move(self, pos):
        """ This method blindly moves a mesh in a given direction. """
        x, y, z = pos
        [v.update(v.x + x, v.y + y, v.z + z) for v in self.vertices]

    
    def rotateNormals(self, angles):
        """ This method rotates the normals of a mesh. """
        xa, ya, za = radians(angles[0]), radians(angles[1]), radians(angles[2])
        xrm, yrm, zrm = (cos(xa),  sin(xa)), (cos(ya),  sin(ya)), (cos(za),  sin(za))
        [rotate_xyz_dummy(face[3], xrm, yrm, zrm) for face in self.polygons]

    def updateNormals(self):
        """ This method updates the normal vectors for each face. """
        index = 0
        while index < len(self.polygons):
            face = self.polygons[index]
            try: 
                face[3].update(getNormal(face[0][0], face[1][0], face[2][0]))
                index += 1
            except:
                del self.polygons[index]
                
            
    def updateBrightness(self):
        """ This method sets the brightness of each face. """
        # starting vector is not (0,0,0) to avoid the occasional 
        # divide by zero error due to rounding errors in the mesh.
        
        normals = [Vect3(0,0.001,0) for _ in range(len(self.vertices))]
        for i in range(len(self.vertices)):
            v = self.vertices[i]
            for face in self.polygons:
                # if the face references the current vertex:
                if v in [face[0][0], face[1][0], face[2][0]]:
                    # add the normal vector for the face to the list of vertex normals
                    normals[i] += face[3]

        # Normalize all vectors:
        [n.normalize() for n in normals]
        
        # Calculate lighting:
        self.v_bright = [-clamp(n.dot(LIGHTING) + LIGHT_BIAS, -1, 1) for n in normals]
        
        # re-assign the index value with brightness. It won't be needed after this.
        for i in range(len(self.polygons)):
            a, b, c = self.polygonI[i]
            a1, b1, c1, n = self.polygons[i]
            a1[2], b1[2], c1[2] = self.v_bright[a], self.v_bright[b], self.v_bright[c]

    def getDistance(self):
        """ This method gets the approximant squared distance from a point to all faces in the mesh. """
        self.depth = [(f[0][0][2] + f[1][0][2] + f[2][0][2]) * ONE_THIRD for f in self.polygons]

    def getFarPoint(self):
        """ This method finds the farthest point from the origin of a mesh. """
        farPoint = max([(v - self.origin).length_squared() for v in self.vertices])
        if farPoint != 0: self.farPoint = sqrt(farPoint)

    def getBounds(self):
        """ This method finds the farthest point from the origin of a mesh. """
        self.bounds = BoundingBox(self.vertices)

    def getCenter(self):
        """ This method gets the center point of a mesh. """
        origin = Vect3(0, 0, 0)
        for v in self.vertices: origin += v
        if len(self.vertices) != 0: 
            origin = origin / len(self.vertices)
        
        self.origin = origin
        print(self.origin)  # debug info.

    def Origin(self):
        return self.origin + self.position

def UnpackMesh(filename,is_static=False, update_lighting=True):
    """ This function loads a .obj file and stores it into a mesh object. .obj files are extremely simple. Each line
    consists of a tag followed by the data for that item. for example, the tag 'v' is for vertex and the following
    information should be three floating point numbers stored in raw text, while the tag 'p' stands for polygon,
    and the information should be 3 integers for the index of each vertex that forms that face. This is done to avoid
    repeating vertices. Seriously, if you want to try something like this on your own just open a .obj in a text
    editor and see how it all goes together. """

    if filename[len(filename) - 4:] not in ['.obj', '.OBJ']:
        raise Exception('file, "'+filename+'" does not end in .obj')

    obj = open(filename, "r")
    vertices, uv_vertices, polygons, uv_polygons = [], [], [], []
    position = Vect3(0, 0, 0)
    
    try:  # Try loading mesh
        for line in obj:
            line = line.strip("\n")
            line = line.split(" ")

            if line[0] == 'v':  # Vertex
                point = Vect3(float(line[1]), float(line[2]), float(line[3]))
                vertices.append(point)

            if line[0] == 'vt':  # UV texture information
                # the V value is inverted because of some bullshit i wrote ages ago.
                # It's dumb but this is the best way I can think of to do this.
                uv = Vect2(float(line[1]), 1 - (float(line[2])))    
                uv_vertices.append(uv)

            elif line[0] == 'f':  # Polygon
                face1, face2, face3 = line[1].split('/'), line[2].split('/'), line[3].split('/')
                polygon = (int(face1[0]) - 1, int(face2[0]) - 1, int(face3[0]) - 1)
                uv_polygon = (int(face1[1]) - 1, int(face2[1]) - 1, int(face3[1]) - 1,)

                polygons.append(polygon)
                uv_polygons.append(uv_polygon)

        # Sneaky work around to get a unique float object for each vertex
        brightness = [0.0 for _ in range(len(vertices))]
        polygonIndex = []

        # I know this looks confusing but I did it this way to keep soft copies of vertices so the .update() method 
        # propagates to the polygons. Basically, I'm storing the reference of each vertex in the polygons list because multiple
        # polygons reference the same vertices so why do the math to translate, rotate, scale, or project them multiple times.
        
        for i in range(len(polygons)):  
            a, b, c = vertices[polygons[i][0]], vertices[polygons[i][1]], vertices[polygons[i][2]]
            u, v, w = uv_vertices[uv_polygons[i][0]], uv_vertices[uv_polygons[i][1]], uv_vertices[uv_polygons[i][2]]
            polygonIndex.append([polygons[i][0], polygons[i][1], polygons[i][2]])
            polygons[i] = [[a, u, 0.0], [b, v, 0.0], [c, w, 0.0], Vect3(0,0,0)]
        
        polygonIndex = tuple(polygonIndex)
        new_mesh = Mesh(polygons, polygonIndex, vertices, brightness, position, is_static, update_lighting)
        
    finally:
        obj.close()

    return new_mesh


def QuickSort(sort, index):
    """ my implementation of the QuickSort algorithm originally written by Tony Hoare, 1960. 
        NOTE: if you want to use the index list to well, index sort, make sure to make a hard copy of 
        sort beforehand. this method destroys the original unsorted list. """

    elements = len(sort)

    # Base case
    if elements < 2:
        return sort, index

    current_position = 0

    for i in range(1, elements):
        if sort[i] < sort[0]:
            current_position += 1
            sort[i], sort[current_position] = sort[current_position], sort[i]
            index[i], index[current_position] = index[current_position], index[i]
    sort[0], sort[current_position], = sort[current_position], sort[0]
    index[0], index[current_position] = index[current_position], index[0]

    # recursively sort blocks
    left = QuickSort(sort[0:current_position], index[0:current_position])
    right = QuickSort(sort[current_position + 1:elements], index[current_position + 1:elements])

    # recombine lists into one list
    return sort, left[1] + [index[current_position]] + right[1]


def getNormal(a, b, c):
    """ This function gets the normal vector of a face. """
    u, v = b - a, c - a
    normal = u.cross(v)
    return normal / normal.length()


class BoundingBox:
    """ This class stores the relevant data for a bounding box. This is used to check if a mesh is visible or not. """
    def __init__(self, vertices):

        # Find the the each axial face of the mesh:
        minX, minY, minZ = min([v.x for v in vertices]), min([v.y for v in vertices]), min([v.z for v in vertices])
        maxX, maxY, maxZ = max([v.x for v in vertices]), max([v.y for v in vertices]), max([v.z for v in vertices])

        # Generate the 8 corner points from the faces:
        vtex  = [Vect3(minX, minY, maxZ), Vect3(minX, maxY, maxZ), Vect3(minX, minY, minZ), Vect3(minX, maxY, minZ), 
                 Vect3(maxX, minY, maxZ), Vect3(maxX, maxY, maxZ), Vect3(maxX, minY, minZ), Vect3(maxX, maxY, minZ)]
        self.vertices = vtex
        
        # Generate the polygons connecting the points
        self.polygons = [(vtex[6], vtex[3], vtex[2]), (vtex[5], vtex[4], vtex[0]), (vtex[5], vtex[1], vtex[3]), 
                         (vtex[0], vtex[4], vtex[6]), (vtex[2], vtex[1], vtex[0]), (vtex[4], vtex[5], vtex[7])] 

    def update(self, vertices):
        """ This method recalculates the bounding box. """
        minX, minY, minZ = min([v.x for v in vertices]), min([v.y for v in vertices]), min([v.z for v in vertices])
        maxX, maxY, maxZ = max([v.x for v in vertices]), max([v.y for v in vertices]), max([v.z for v in vertices])

        new_vertices  = (Vect3(minX, minY, maxZ), Vect3(minX, maxY, maxZ), Vect3(minX, minY, minZ), Vect3(minX, maxY, minZ), 
                         Vect3(maxX, minY, maxZ), Vect3(maxX, maxY, maxZ), Vect3(maxX, minY, minZ), Vect3(maxX, maxY, minZ))

        [old_vertex.update(new_vertex) for old_vertex, new_vertex in zip(self.vertices, new_vertices)]


def meshBoundingSphereCull(camera, origin, radius):
        """ This method checks if a mesh can be seen by the camera. Result is approximate but very quick. """
        # Get position relative to the camera:
        point = origin - camera.position
        point.rotate_y_ip(-camera.y_rotation)
        point.rotate_x_ip(camera.x_rotation)
        return not (True in [plane.pointToPlane(point) < -radius for plane in camera.clip])

def meshBoundingBoxCull(camera, poly):
    """ This function checks if a bounding box can be seen by the camera. this uses a modified 
    version of the algorithm used in ClipMesh. """
    
    # Iterate through the planes, discard faces until there are none left, or all planes are checked.
    for plane in camera.r_clip:
        pd, pn, = plane.d, plane.n
    
        # Remove or clip faces to the camera's view.
        index = 0
        while index < len(poly):
            a, b, c = poly[index]
            # Determine point-to-plane-distance for a, b, and c.
            a_inside, b_inside, c_inside = pn.dot(a) - pd > 0, pn.dot(b) - pd > 0, pn.dot(c) - pd > 0
            inside = a_inside + b_inside + c_inside
            if inside == 0:  # Face is offscreen, remove from list.
                del poly[index]
            else:
                index += 1

    return len(poly) != 0

def clipMesh(clip, mesh_object):
    """ This method clips a mesh against the camera's view. """
    
    mesh, vtex = mesh_object.polygons, mesh_object.vertices

    # clip mesh against all planes:

    for plane in clip:
        pd, pn, = plane.d, plane.n

        # Remove invalid vertices from vertex list, leaving the copy found in mesh as the only one remaining.
        to_remove = [v for v in vtex if not pn.dot(v) - pd > -0.0001]
        [vtex.remove(v) for v in to_remove]

        # Remove or clip faces to the camera's view.
        inside = 0
        index = 0
        while True:

            if inside != 0: # if the last face was discarded, don't increment.
                index += 1
            
            if index >= len(mesh): # Exit the loop if at the end.
                break

            a, b, c, n = mesh[index]
            # Determine point-to-plane-distance for a, b, and c.
            a_inside = pn.dot(a[0]) - pd > -0.0001
            b_inside = pn.dot(b[0]) - pd > -0.0001
            c_inside = pn.dot(c[0]) - pd > -0.0001
            inside = a_inside + b_inside + c_inside

            if inside == 3:     # face is on screen. Ignore it.
                continue
            
            elif inside == 0:   # Face is offscreen, remove from list.
                del mesh[index]
                continue

            elif inside == 1:       # Two points off-screen, clip into trigon, update face.
                if a_inside:
                    b, c = plane.vertexPlaneIntersect(a, b), plane.vertexPlaneIntersect(a, c)
                    vtex.extend([b[0], c[0]])
                    mesh[index][1:3] = b, c
                    continue
                elif b_inside:
                    a, c = plane.vertexPlaneIntersect(b, a), plane.vertexPlaneIntersect(b, c)
                    vtex.extend([a[0], c[0]])
                    mesh[index][0], mesh[index][2] = a, c
                    continue
                else:
                    b, a = plane.vertexPlaneIntersect(c, b), plane.vertexPlaneIntersect(c, a)
                    vtex.extend([b[0], a[0]])
                    mesh[index][:2] = a, b
                    continue

            elif inside == 2:       # One point off-screen. Clip into quad then trigon, update and append face.
                if not a_inside:  # A is off-screen
                    ab, ac = plane.vertexPlaneIntersect(a, b), plane.vertexPlaneIntersect(a, c)
                    vtex.extend([ab[0], ac[0]])
                    if (ac[0] - b[0]).magnitude_squared() < (ab[0] - c[0]).magnitude_squared():
                        mesh[index][:3] = b, ac, ab
                        mesh.append([b, c, ac, n])
                        continue
                    else:
                        mesh[index][:3] = c, ab, b
                        mesh.append([c, ac, ab, n])
                        continue
                elif not b_inside:  # B is off-screen
                    bc, ba = plane.vertexPlaneIntersect(b, c), plane.vertexPlaneIntersect(b, a)
                    vtex.extend([bc[0], ba[0]])
                    if (ba[0] - c[0]).magnitude_squared() < (bc[0] - a[0]).magnitude_squared():
                        mesh[index][:3] = c, ba, bc
                        mesh.append([c, a, ba, n])
                        continue
                    else:
                        mesh[index][:3] = a, bc, c
                        mesh.append([a, ba, bc, n])
                        continue
                else:  # C is off-screen
                    cb, ca = plane.vertexPlaneIntersect(c, b), plane.vertexPlaneIntersect(c, a)
                    vtex.extend([cb[0], ca[0]])
                    if (cb[0] - a[0]).magnitude_squared() < (ca[0] - b[0]).magnitude_squared():
                        mesh[index][:3] = a, cb, ca
                        mesh.append([a, b, cb, n])
                        continue
                    else:
                        mesh[index][:3] = b, cb, ca
                        mesh.append([b, ca, a, n])
                        continue
            else:
                print("aww fuck.")

def fragMesh(mesh, step=8):
    """ This function breaks a static mesh up into fragments for faster render times. """

    vtex, poly, polyindex = mesh.vertices, mesh.polygons, mesh.polygonI

    meshes = []

    # Find the bounds of the mesh:
    minX = floor(min([v.x for v in vtex])) - step
    maxX = ceil(max([v.x for v in vtex])) + step

    minY = floor(min([v.y for v in vtex])) - step
    maxY = ceil(max([v.y for v in vtex])) + step

    minZ = floor(min([v.z for v in vtex])) - step
    maxZ = ceil(max([v.z for v in vtex])) + step

    # Find the subdivision distance

    for x in range(minX, maxX, step):
        for y in range(minY, maxY, step):
            for z in range(minZ, maxZ, step):
                newpoly, newvtex, newlght, index, currentstep = [], [], [], 0, Vect3(x,y,z)
                nextstep = currentstep + Vect3(step)
                while index < len(poly):
                    face = poly[index]
                    for vertex in face[:3]:
                        p = vertex[0]
                        # Check if the current point is in bounds:
                        if currentstep.elementwise() < p.elementwise() < nextstep.elementwise():
                            a, b, c = face[:3]
                            del poly[index]
                            newpoly.append(face)
                            if a[0] not in newvtex: 
                                newvtex.append(a[0])
                                newlght.append(a[2])
                            if b[0] not in newvtex: 
                                newvtex.append(b[0])
                                newlght.append(a[2])
                            if c[0] not in newvtex: 
                                newvtex.append(c[0])
                                newlght.append(a[2])
                            index -= 1
                            break
                    
                    index += 1 
                    
                if len(newpoly) != 0:
                    newMesh = Mesh(newpoly, [], newvtex, newlght, Vect3(), True, False, mesh.texIndex)
                    meshes.append(dc(newMesh))

    return meshes
