# This program was writen by Ethan Parker
# This program consolidates all the mathematics used in rendering.

from math import *
from constants import *
from pygame import Vector3 as Vect3
from pygame import Vector2 as Vect2
from copy import deepcopy as dc


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


def cosineSolve(v1, v2):
    a, b, c = v2.length(), v1.length(), (v1 - v2).length()
    try:
        return acos(((a*a)+(b*b)-(c*c)) / (2*a*b))
    except (ZeroDivisionError, ValueError):
        return 0


def AngleTo(a, b=Vect3(0,0,1)):
    """ This function determins the angle of a unit vector. """
    x = cosineSolve(Vect2(a.y, a.z), Vect2(b.y, b.z)) * 57.2958
    z = cosineSolve(Vect2(a.x, a.y), Vect2(b.x, b.y)) * 57.2958
    y = cosineSolve(Vect2(a.x, a.z), Vect2(b.x, b.z)) * 57.2958
    return [x, y, z]


def Bezier(line, t):
    """ This function calcualtes a bezier curve recersively. """
    if len(line) == 1: return line[0]
    else: return (1 - t) * Bezier(line[0:-1], t) + t * Bezier(line[1:], t)


def smoothLerp(a, b, t):
    """ Returns the smoothly interpolated value between a and b at t in range [0,1]. """
    return Lerp(a, b, Lerp(t*t, 1-((t-1)*(t-1)), t))


def Lerp(a, b, t):
    """ Returns the interpolated value between a and b at t in range [0,1]."""
    return a + (b - a) * t    


def pointOnTrigon(point, a, b, c):
    """ This function handles checking of a Vect3 is intersecting with a Trigon. """

    # Offset points by origin:
    a, b, c = a - point, b - point, c - point
    # Get normals of faces formed by a, b, c to p:
    u, v, w = b.cross(c), c.cross(a), a.cross(b)

    # if the normals are not parrelel, the point does not interesct the face.
    if u.dot(v) < 0 or u.dot(w) < 0:
        return False

    return True


class Plane:
    """ This class defines a basic 3d plane as defined by 3 points. """
    def __init__(self, a, b, c):
        self.n = getNormal(a, b, c)
        self.p = (a + b + c) * 0.333333332
        self.d = self.p.dot(self.n)

    def rotate_x(self, angle):
        """ This method rotates the plane around the X axis. """
        angle = radians(angle)
        cos_sin = (cos(angle), sin(angle))
        rotate_x_dummy(self.n, cos_sin)
        rotate_x_dummy(self.p, cos_sin)
    
    def rotate_y(self, angle):
        """ This method rotates the plane around the Y axis. """
        angle = radians(angle)
        cos_sin = (cos(angle), sin(angle))
        rotate_y_dummy(self.n, cos_sin)
        rotate_y_dummy(self.p, cos_sin)

    def rotate_z(self, angle):
        """ This method rotates the plane around the Z axis. """
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
        t = (self.d - ad) / ((end.dot(self.n)) - ad)
        return ((end - start) * t) + start

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
    def __init__(self, polygons=[], polygonIndex=(), vertices=[], brightness=[], position=Vect3(0, 0, 0), texIndex=0, static=False, updatelighting=True):
        """ Vertices must be a list containing Vect3 [point 1, point 2, point 3 ...]
            Polygons must be a list containing lists containing soft copies of vertices forming a face: [[1,2,3], ...]. """

        self.vertices = vertices        # List of vertices
        self.polygons = polygons        # List of faces connecting vertices
        self.polygonI = polygonIndex    # List containing the index of a vertex in the verices list
        self.v_bright = brightness      # List of brightnesses for each vertex
        self.farPoint = 0               # farthest point on the object
        self.texIndex = texIndex        # Stores the texture index for this mesh
        self.depth = []                 # List containing distance from camera for each face on a mesh
        self.rotation = [0, 0, 0]       # Rotational offset
        self.position = position        # Positional offset
        self.origin = position          # Center point of the mesh
        self.facing = Vect3(0, 0, 1)    # Forward direction of mesh

        self.static = static            # Bool for if mesh is moveable.
        
        # Run initialization routines. 
        self.updateNormals()
        self.getCenter()
        self.getFarPoint()
        
        if updatelighting: 
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
            # Move to the target - the current possition. (same as move to 0, then to target)
            self.position = pos
            x, y, z = pos
            [v.update(v.x + x, v.y + y, v.z + z) for v in self.vertices]

    def move(self, pos):
        """ This method blindly moves a mesh in a given direction. """
        x, y, z = pos
        [v.update(v.x + x, v.y + y, v.z + z) for v in self.vertices]

    def staticMove(self, pos):
        """ This function applys a translation to a mesh set to Static. """
        if self.static:
            self.position = pos
            x, y, z = pos
            [v.update(v.x + x, v.y + y, v.z + z) for v in self.vertices]

    def rotateNormals(self, angles):
        """ This method rotates the normals of a mesh. """
        xa, ya, za = radians(angles[0]), radians(angles[1]), radians(angles[2])
        xrm, yrm, zrm = (cos(xa),  sin(xa)), (cos(ya),  sin(ya)), (cos(za),  sin(za))
        [rotate_xyz_dummy(face[3], xrm, yrm, zrm) for face in self.polygons]

    def updateNormals(self):
        """ This method updates the normal vectors for each face. """
        for face in self.polygons:
            face[3].update(getNormal(face[0][0], face[1][0], face[2][0]))
            
    def updateBrightness(self):
        """ This method sets the brightness of each face. """
        # starting vector is not (0,0,0) to avoid the ocasional 
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
        self.v_bright = [(n.dot(LIGHTING) * MAXLIGHTING) + LIGHTINGBIAS for n in normals]
        
        # reasign the index value with brightness. It won't be needed after this.
        for i in range(len(self.polygons)):
            a, b, c = self.polygonI[i]
            a1, b1, c1, n = self.polygons[i]
            a1[2], b1[2], c1[2] = self.v_bright[a], self.v_bright[b], self.v_bright[c]

    def getDistance(self):
        """ This method gets the aproximent squared distance from a point to all faces in the mesh. """
        self.depth = [(f[0][0][2] + f[1][0][2] + f[2][0][2]) * ONETHIRD for f in self.polygons]

    def getFarPoint(self):
        """ This method finds the farthest point from the origin of a mesh. """
        farPoint = max([(v - self.origin).length_squared() for v in self.vertices])
        if farPoint != 0: self.farPoint = sqrt(farPoint)

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


def UnpackMesh(filename):
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
                uv = Vect2(float(line[1]), float(line[2]))
                uv_vertices.append(uv)

            elif line[0] == 'f':  # Polygon
                face1, face2, face3 = line[1].split('/'), line[2].split('/'), line[3].split('/')
                polygon = (int(face1[0]) - 1, int(face2[0]) - 1, int(face3[0]) - 1)
                uv_polygon = (int(face1[1]) - 1, int(face2[1]) - 1, int(face3[1]) - 1,)

                polygons.append(polygon)
                uv_polygons.append(uv_polygon)

        # Sneaky work around to get a unique float for each vertex
        brightness = [0.0 for _ in range(len(vertices))]
        polygonIndex = []

        for i in range(len(polygons)):
            a, b, c = vertices[polygons[i][0]], vertices[polygons[i][1]], vertices[polygons[i][2]]
            u, v, w = uv_vertices[uv_polygons[i][0]], uv_vertices[uv_polygons[i][1]], uv_vertices[uv_polygons[i][2]]
            polygonIndex.append([polygons[i][0], polygons[i][1], polygons[i][2]])
            polygons[i] = [[a, u, 0.0], [b, v, 0.0], [c, w, 0.0], Vect3(0, 0, 0)]

        polygonIndex = tuple(polygonIndex)

        new_mesh = Mesh(polygons, polygonIndex, vertices, brightness, position)
        
    finally:
        obj.close()

    return new_mesh


def QuickSort(sort, index):
    """my implementation of the QuickSort algorithm originally writen by Tony Hoare, 1960. """

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
    if normal[:] != [0, 0, 0]:
        return normal / normal.length()
    return normal


def clipMesh(clip, mesh_object):
    """ This method clips a mesh against the camera's view. """
    
    mesh, vtex = mesh_object.polygons, mesh_object.vertices

    # clip mesh against all planes:

    for plane in clip:
        pd, pn, = plane.d, plane.n

        # Remove invalid vertices from vertex list, leaving the copy found in mesh as the only one remaining.
        removed = [v for v in vtex if not pn.dot(v) - pd > -0.0001]
        [vtex.remove(v) for v in removed]

        # Remove or clip faces to the camera's view.
        index = 0
        while index < len(mesh):
            a, b, c, n = mesh[index]
            # Determin point-to-plane-distance for a, b, and c.
            a_inside = pn.dot(a[0]) - pd > -0.0001
            b_inside = pn.dot(b[0]) - pd > -0.0001
            c_inside = pn.dot(c[0]) - pd > -0.0001
            inside = a_inside + b_inside + c_inside
            if inside == 0:         # Face is offscreen, remove from list.
                del mesh[index]
                index -= 1
            elif inside == 1:       # Two points off-screen, clip into trigon, update face.
                if a_inside:
                    b, c = plane.vertexPlaneIntersect(a, b), plane.vertexPlaneIntersect(a, c)
                    vtex.extend([b[0], c[0]])
                elif b_inside:
                    a, c = plane.vertexPlaneIntersect(b, a), plane.vertexPlaneIntersect(b, c)
                    vtex.extend([a[0], c[0]])
                else:
                    b, a = plane.vertexPlaneIntersect(c, b), plane.vertexPlaneIntersect(c, a)
                    vtex.extend([b[0], a[0]])
                mesh[index][:3] = a, b, c

            elif inside == 2:       # One point off-screen. Clip into quad then trigon, update and append face.
                if not a_inside:  # A is off-screen
                    ab, ac = plane.vertexPlaneIntersect(a, b), plane.vertexPlaneIntersect(a, c)
                    vtex.extend([ab[0], ac[0]])
                    mesh[index][:3] = c, b, ac
                    mesh.append([b, ab, ac, n])

                elif not b_inside:  # B is off-screen
                    bc, ba = plane.vertexPlaneIntersect(b, c), plane.vertexPlaneIntersect(b, a)
                    vtex.extend([bc[0], ba[0]])
                    mesh[index][:3] = a, c, bc
                    mesh.append([a, ba, bc, n])

                else:  # C is off-screen
                    cb, ca = plane.vertexPlaneIntersect(c, b), plane.vertexPlaneIntersect(c, a)
                    vtex.extend([cb[0], ca[0]])
                    mesh[index][:3] = b, a, ca
                    mesh.append([b, cb, ca, n])
            index += 1

    return mesh_object


def fragMesh(mesh,step=8):
    """ This function breaks a static mesh up into fragments for faster render times. """

    vtex, poly, polyindex = mesh.vertices, mesh.polygons, mesh.polygonI

    meshes = []

    # Find the bounds of the mesh:
    minX = int(min([v.x for v in vtex])) - step
    maxX = int(max([v.x for v in vtex])) + step

    minY = int(min([v.y for v in vtex])) - step
    maxY = int(max([v.y for v in vtex])) + step

    minZ = int(min([v.z for v in vtex])) - step
    maxZ = int(max([v.z for v in vtex])) + step

    # Find the subdivision distance

    for x in range(minX, maxX, step):
        for y in range(minY, maxY, step):
            for z in range(minZ, maxZ, step):
                newpoly = []
                newvtex = []
                newlght = []
                newpolyindex = []
                index = 0
                while index < len(poly):
                    face = poly[index]
                    faceindex = polyindex[index]
                    a, b, c = face[:3]
                    
                    for vertex in face[:3]:
                        p = vertex[0]
                        # Check if the current point is in bounds:
                        if x <= p.x <= x + step and y <= p.y <= y + step and z <= p.z <= z + step:
                            del poly[index]
                            newpoly.append(face)
                            newpolyindex.append(faceindex)
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
                    meshes.append(Mesh(newpoly, newpolyindex, newvtex, newlght, Vect3(0, 0, 0), mesh.texIndex, True, False))

    return meshes


