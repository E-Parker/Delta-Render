# This program contains miscellaneous functions for version 2.4 of ΔRender
# This program was writen by Ethan Parker

import copy
import math
import pygame
# import os
from pygame.math import Vector2 as Vect2
from pygame.math import Vector3 as Vect3

# import make


class Vertex:
    """ This class contains a point in 3d space and texture information."""
    def __init__(self, point=None, uv=None):
        self.p = point
        self.t = uv

    def interp_x(self, b, y):
        """ This function interpolates a Vertex to a given x value."""
        point = Vect3(interp(self.p.x, self.p.y, b.p.x, b.p.y, y), y, interp(self.p.z, self.p.y, b.p.z, b.p.y, y))
        tx_point = Vect2(interp(self.t.x, self.p.y, b.t.x, b.p.y, y), interp(self.t.y, self.p.y, b.t.y, b.p.y, y))
        self.p, self.t = point, tx_point

    def interp_y(self, b, y):
        """ This function interpolates a Vertex to a given y value."""
        point = Vect3(interp(self.p.x, self.p.y, b.p.x, b.p.y, y), y, interp(self.p.z, self.p.y, b.p.z, b.p.y, y))
        tx_point = Vect2(interp(self.t.x, self.p.y, b.t.x, b.p.y, y), interp(self.t.y, self.p.y, b.t.y, b.p.y, y))
        self.p, self.t = point, tx_point

    def interp_z(self, b, z):
        """ This function interpolates a Vertex to a given z value."""
        point = Vect3(interp(self.p.x, self.p.y, b.p.x, b.p.y, z), interp(self.p.y, self.p.z, b.p.y, b.p.z, z), z)
        tx_point = Vect2(interp(self.t.x, self.p.z, b.t.x, b.p.z, z), interp(self.t.y, self.p.z, b.t.y, b.p.z, z))
        self.p, self.t = point, tx_point

    def lerp(self, b, amount):
        """ This function handles interpolating between two vertices. """
        new_point = self.p.lerp(b.point, amount)
        new_uv = self.t.lerp(b.uv, amount)

        return Vertex(new_point, new_uv)


class Trigon:
    """ This class represents a simple 3 sided polygon with texture coordinates.
        Provide a list of vertices (vect3), index of vertices forming a trigon [A, B, C],
        a list of texture coordinates (vect2), index of texture coordinates [U, V, W]. 
        Manual assignment is also a valid option."""
    
    def __init__(self, vertices=None, polygon=None, uv=None, uv_polygon=None):
        # Variables:
        try:
            self.a = Vertex(vertices[polygon[0]], uv[uv_polygon[0]])
            self.b = Vertex(vertices[polygon[1]], uv[uv_polygon[1]])
            self.c = Vertex(vertices[polygon[2]], uv[uv_polygon[2]])

            # self.n = getNormal(self.a, self.b, self.c)
            self.n = None
        
        except TypeError:
            self.a = Vertex()
            self.b = Vertex()
            self.c = Vertex()

    def __iter__(self):
        """ This function handles indexing a Trigon as if it were a list. """
        yield self.a
        yield self.b
        yield self.c

    def update(self, a, b, c):
        """ This function updates all points of the trigon. """
        self.a, self.b, self.c = a, b, c


class Mesh:
    """ This class stores a 3D object consisting of vertices and polygons that connect said vertices.
        The structure of this object is identical to the .obj file format. """
    def __init__(self, vertices=None, polygons=None, uv_vertices=None, uv_polygons=None):
        """ Vertices must be a list containing Vect3 [point 1, point 2, point 3 ...]
            Polygons must be a list containing lists containing index of vertices forming a face: [[0,1,2], ...].
            Data structure is the same for uv_vertices, and uv_polygons, just with Vect2 for texture coordinates."""

        if vertices is None:
            vertices = ()
        if polygons is None:
            polygons = ()
        if uv_vertices is None:
            uv_vertices = ()
        if uv_polygons is None:
            uv_polygons = ()

        self.vertices = vertices        # List of vertices
        self.polygons = polygons        # List of faces connecting vertices
        self.depth = []                 # Average depth information for each face
        self.uv_vertices = uv_vertices  # UV Texture coordinates
        self.uv_polygons = uv_polygons  # Faces connecting UV
        self.texture = None             # Assign texture later.

        self.rotation_vel = [0, 0, 0]   # Current rotational velocity
        self.rotation = [0, 0, 0]       # Rotational offset
        self.position = Vect3(0, 0, 0)  # Positional offset
        self.facing = Vect3(0, 0, 1)    # Forward direction of mesh
    
    def __len__(self):
        """ Returns the number of polygons in the mesh. """
        return len(self.polygons)

    def __getitem__(self, key):
        """ Return the polygon formed by vertices at index."""
        try:
            item = Trigon(self.vertices, self.polygons[key], self.uv_vertices, self.uv_polygons[key])
            return item
        except IndexError:
            warn(1, key)
            return

    def __delitem__(self, key):
        if key != len(self.polygons):
            self.polygons = self.polygons[:key] + self.polygons[key + 1:]
            self.uv_polygons = self.uv_polygons[:key] + self.uv_polygons[key + 1:]
        else:

            self.polygons = self.polygons[:key]
            self.uv_polygons = self.uv_polygons[:key]

    def __setitem__(self, key, trigon):
        """ This function handles adding a new trigon to the mesh. """
        self.setItem(key, trigon)

    def append(self, trigon):
        """ This function handles appending a new trigon to the mesh. """
        self.setItem(None, trigon)

    def setItem(self, key, trigon):
        """ This function handles setting or adding a polygon to the mesh.
            Used in both __setitem__ and append methods. """
        a_index = self.setVertex(trigon.a.p)
        b_index = self.setVertex(trigon.b.p)
        c_index = self.setVertex(trigon.c.p)

        u_index = self.setTextureVertex(trigon.a.t)
        v_index = self.setTextureVertex(trigon.b.t)
        w_index = self.setTextureVertex(trigon.c.t)

        if key is None:
            self.polygons += ((a_index, b_index, c_index),)
            self.uv_polygons += ((u_index, v_index, w_index),)
        else:
            self.polygons = self.polygons[:key] + ((a_index, b_index, c_index),) + self.polygons[key + 1:]
            self.uv_polygons = self.uv_polygons[:key] + ((u_index, v_index, w_index),) + self.uv_polygons[key + 1:]

    def rotate_x(self, angle):
        """ Update x-axis rotation of all points."""
        self.rotation_vel[0] = angle
        self.facing.rotate_x(angle)

        iterator = map(self.vertex_rotate_x, self.vertices)
        self.vertices = tuple(iterator)

    def rotate_y(self, angle):
        """ Update y-axis rotation of all points."""
        self.rotation_vel[1] = angle
        self.facing.rotate_y(angle)

        iterator = map(self.vertex_rotate_y, self.vertices)
        self.vertices = tuple(iterator)

    def rotate_z(self, angle):
        """ Update z-axis rotation of all points."""
        self.rotation_vel[2] = angle
        self.facing.rotate_z(angle)

        iterator = map(self.vertex_rotate_z, self.vertices)
        self.vertices = tuple(iterator)

    def look_at(self, vector):
        """ Point mesh in direction of vector. """
        x_angle = (Vect2(self.facing.x, self.facing.y)).angle_to(Vect2(vector.x, vector.y))
        y_angle = (Vect2(self.facing.x, self.facing.z)).angle_to(Vect2(vector.x, vector.z))
        z_angle = (Vect2(self.facing.y, self.facing.z)).angle_to(Vect2(vector.y, vector.z))

        self.rotate_x(x_angle)
        self.rotate_y(y_angle)
        self.rotate_z(z_angle)

    def move_to(self, position):
        if self.position != position:
            self.position = -1 * self.position
            iterator = map(self.vertex_move, self.vertices)
            self.vertices = tuple(iterator)

            self.position = position
            iterator = map(self.vertex_move, self.vertices)
            self.vertices = tuple(iterator)

    def move(self, position):
        self.position = position
        iterator = map(self.vertex_move, self.vertices)
        self.vertices = tuple(iterator)
        self.position = Vect3(0, 0, 0)

    def update(self):
        """ Reset average depth for sorting later. """
        if self.rotation_vel != 0:
            self.rotation += self.rotation_vel

        iterator = map(self.average_depth, self.polygons)
        self.depth = list(iterator)

    def setVertex(self, item):
        """ This function adds a single vertex to the list of vertices and returns its location in said list.
            Note: A vertex is only added if it does not already exist in the list of vertices."""
        if item in self.vertices:
            index = self.vertices.index(item)
        else:
            self.vertices += (item,)
            index = len(self.vertices) - 1

        return index

    def setTextureVertex(self, item):
        """ This function adds a single texture coordinate to the list of uv_vertices. """
        if item in self.uv_vertices:
            index = self.uv_vertices.index(item)
        else:
            self.uv_vertices += (item,)
            index = len(self.uv_vertices) - 1

        return index

    def vertex_rotate_x(self, vertex):
        return (vertex - self.position).rotate_x(self.rotation_vel[0]) + self.position

    def vertex_rotate_y(self, vertex):
        return (vertex - self.position).rotate_y(self.rotation_vel[1]) + self.position

    def vertex_rotate_z(self, vertex):
        return (vertex - self.position).rotate_z(self.rotation_vel[2]) + self.position

    def vertex_move(self, vertex):
        return vertex + self.position

    def average_depth(self, face):
        return (self.vertices[face[0]].z + self.vertices[face[1]].z + self.vertices[face[2]].z) / 3


class Plane:
    """ This class defines a basic 3d plane as defined by 3 points. """
    def __init__(self, a, b, c):

        self.n = getNormal(a, b, c)
        self.p = (a + b + c) / 3
        self.d = self.p.dot(self.n)

    def pointToPlane(self, p):
        """ This function calculates the point-to-plane distance from any given point. """
        distance = self.n.dot(p) - self.d
        return distance

    def vectorPlaneIntersect(self, start, end):
        """ This function calculates the intersection point of a ray and a plane. """
        ad = start.dot(self.n)
        t = (self.d - ad) / ((end.dot(self.n)) - ad)
        ray = end - start
        intersect = (ray * t) + start
        return intersect

    def vertexPlaneIntersect(self, start, end):
        """ This function calculates the intersection point of a vertex with texture coordinate and a plane. """
        ad = start.p.dot(self.n)
        t = (self.d - ad) / ((end.p.dot(self.n)) - ad)

        intersect = ((end.p - start.p) * t) + start.p
        tx_intersect = ((end.t - start.t) * t) + start.t

        new_vertex = Vertex(intersect, tx_intersect)
        return new_vertex


class Camera:
    """ This class is a generic camera class used for rendering. """
    def __init__(self, fov, near, far, resolution, size):
        
        # Variables
        self.fov = fov
        self.deviation = (self.fov / 360) / 4  # used checking of an object can be seen by the camera.
        self.resolution = resolution
        self.size = size
        self.near = near
        self.far = far
        self.x_rotation = 0
        self.y_rotation = 0
        self.sensitivity = 2.5 / 1000
        self.radius = 0.5
        self.height = 1
        self.gravity = Gravity()
        self.clip = self.genClip()

        # Movement:
        self.move = False
        self.move_w = False
        self.move_e = False
        self.move_n = False
        self.move_s = False

        self.movespeed = 0
        self.maxspeed = 0.025

        # Vectors
        self.position = Vect3(0, 0, 0)
        self.rotation = Vect3(0, 0, 1)
        self.velocity = Vect3(0, -1, 0)

    def genClip(self):
        """ This function handles generating the points that form the clipping plane used when rendering. """

        # Solve for horizontal and vertical fov in radians
        horizontal_fov = (500 - 11.5) * 0.01745329251
        vertical_fov = horizontal_fov * 1.77

        print(self.fov)
        print(horizontal_fov, vertical_fov)

        # Solve for the top and left sides of the far plane
        horizontal_far_side_length = self.far * (math.tan(horizontal_fov))  # solve for horizontal side length
        vertical_far_side_length = self.far * (math.tan(vertical_fov))      # solve for vertical side length

        print(horizontal_far_side_length, vertical_far_side_length)

        # generate points:
        c1 = Vect3(horizontal_far_side_length, vertical_far_side_length, self.far)
        c2 = Vect3(horizontal_far_side_length, -1 * vertical_far_side_length, self.far)
        c3 = c1.normalize() * self.near
        c4 = c2.normalize() * self.near
        c5 = Vect3(-1 * horizontal_far_side_length, vertical_far_side_length, self.far)
        c6 = Vect3(-1 * horizontal_far_side_length, -1 * vertical_far_side_length, self.far)
        c7 = c5.normalize() * self.near
        c8 = c6.normalize() * self.near

        # Generate planes from points.
        planes = (Plane(c1, c5, c7), Plane(c6, c2, c4), Plane(c6, c5, c1), Plane(c7, c4, c3), Plane(c3, c2, c1),
                  Plane(c5, c6, c8))

        return planes

    def update(self, colliders, frame_delta, center):
        """ This function handles updating the camera's position, rotation and velocity.
            The value returned by this function determines the game state."""
        # handle inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                # update the direction of the player
                if event.key == pygame.K_ESCAPE: return False
                if event.key == pygame.K_SPACE: self.gravity.set_jumping()
                if event.key == ord('a'): self.move_w, self.move_e, self.move = True, False, True
                if event.key == ord('d'): self.move_e, self.move_w, self.move = True, False, True
                if event.key == ord('w'): self.move_n, self.move_s, self.move = True, False, True
                if event.key == ord('s'): self.move_s, self.move_n, self.move = True, False, True
            elif event.type == pygame.KEYUP:
                # the player has stopped moving
                if event.key == ord('a'): self.move_w = False
                if event.key == ord('d'): self.move_e = False
                if event.key == ord('w'): self.move_n = False
                if event.key == ord('s'): self.move_s = False

        # if none of the directional keys are being pressed, the player isn't moving.
        if self.move_w or self.move_e or self.move_n or self.move_s:
            self.move = True
        else:
            self.move = False

        # Update Vectors:
        self.movespeed = self.maxspeed * frame_delta
        if self.move:
            self.velocity = Vect3(0, 0, 1).rotate_y(-self.y_rotation) * self.movespeed
        else:
            self.velocity = Vect3(0, 0, 0)
        new_vector = Vect3(0, 0, 0)

        if self.move_n: new_vector += self.velocity
        if self.move_s: new_vector += self.velocity.rotate_y(180)
        if self.move_w: new_vector += self.velocity.rotate_y(90)
        if self.move_e: new_vector += self.velocity.rotate_y(270)

        # Calculate Gravity:
        self.gravity.update(frame_delta)
        gravity = Vect3(0, 0.05 * self.gravity.current_value, 0) * frame_delta
        new_vector += gravity

        # Check for collisions
        collisions = 0
        for collider in colliders:
            if collider.enabled:
                collision = collider.sphereCollideCheck(self.position, new_vector, self.radius)
                new_vector += collision[0]
                collisions += collision[1]
        if collisions > 0:
            self.gravity.set_falling()

        # I should add a ray cast from previous position to new position and check for missed collisions.
        # If the player had a new position, "n", but the ray from the previous position to "n" intersects a wall "i",
        # I would just move the player to "i" and run the collision detect again.

        # Update position:
        self.position += new_vector

        # Get mouse position
        mouse_position = pygame.mouse.get_pos()
        pygame.mouse.set_pos([center[0], center[1]])
        roty = (mouse_position[0] - center[0]) * self.sensitivity
        rotx = (mouse_position[1] - center[1]) * self.sensitivity

        self.x_rotation += rotx * 57.2957795131
        self.y_rotation += roty * 57.2957795131

        self.rotation.rotate_x_rad(rotx)
        self.rotation.rotate_y_rad(roty)

        self.rotation.normalize()  # Insure rotation vector is a unit vector.
        return True


class Gravity:
    """ This function approximates gravity. """
    def __init__(self):
        self.amplitude = -0.03125
        self.offset = 1
        self.current_time = 0
        self.current_value = 0

    def set_falling(self):
        if self.current_time < 0:
            self.current_time = 0

    def set_jumping(self):
        self.current_time = -4

    def update(self, frame_delta):
        self.current_time += frame_delta * 0.5

        if self.current_time > 8:
            self.current_time = 8

        self.current_value = self.amplitude * (self.current_time * self.current_time) + self.offset


class Collider:
    """ This class stores all the relevant information for a mesh collider,
    along with methods to collide with objects. """

    def __init__(self, mesh, gap, enabled):

        self.mesh = mesh
        self.planes = ()
        self.points = ()
        self.enabled = enabled

        for i in range(len(self.mesh)):

            face = self.mesh[i]

            # Generate plane for each face so speed up runtime execution:
            a, b, c = face.a.p, face.b.p, face.c.p
            plane = Plane(a, b, c)
            self.planes += (plane,)

            # Generate list of points that form the face:
            new_points = (a, b, c,)
            y_step = int(a.distance_to(c) / gap)

            for y in range(y_step):
                y_interp = (y / y_step)

                cA = a.lerp(c, y_interp)
                cB = b.lerp(c, y_interp)

                # Get the distance between the newly generated points from previous lerp operation:
                x_step = int(cA.distance_to(cB) / gap)

                for x in range(x_step):
                    x_interp = (x / x_step)
                    new_point = cA.lerp(cB, x_interp)
                    new_points += (new_point,)

            # add new points to temporary container.
            self.points += (new_points,)

    def sphereIntersect(self, pos, rad):
        """ This function handles checking for intersections with a sphere collider."""
        # this method is functionally the same as the other method that unintersects the sphere.
        # this one is more efficient for triggers / non-colliding colliders.
        radSqr = rad * rad
        for i in range(len(self.points)):
            distance = self.planes[i].pointToPlane(pos)
            if rad > distance > -rad:
                for point in self.points[i]:
                    if ((point - pos) * (point - pos)) < radSqr:
                        return True
        return False

    def sphereCollideCheck(self, pos, vel, radius):
        """ This function handles un-intersecting a sphere collider with a mesh."""

        # delta_shift   -       Vector3 that will un-intersect the player from all of the faces collided wit
        # collisions    -       number of collisions detected. used when calculating delta shift
        # distance      -       stores point-to-plane distance.

        delta_shift = pygame.math.Vector3(0, 0, 0)
        collisions = 0
        radius_squared = radius * radius
        new_pos = pos + vel

        # Check for collision:
        for i in range(len(self.points)):
            distance = self.planes[i].pointToPlane(new_pos)
            if radius > distance > -radius:
                for point in self.points[i]:
                    if ((point - new_pos) * (point - new_pos)) < radius_squared:
                        collisions += 1
                        if self.planes[i].n[1] > 0.6:
                            delta_shift += pygame.math.Vector3(0, radius - distance, 0)
                        else:
                            delta_shift += pygame.math.Vector3(self.planes[i].n * (radius - distance))
                        break
        if collisions != 0:
            delta_shift = delta_shift / collisions
        return delta_shift, collisions


def getNormal(a, b, c):
    u, v = b - a, c - a

    normal = Vect3((u[1] * v[2]) - (u[2] * v[1]),
                   (u[2] * v[0]) - (u[0] * v[2]),
                   (u[0] * v[1]) - (u[1] * v[0]))

    if normal != (0, 0, 0):
        normal = normal.normalize()

    return normal


def getNormals(mesh):
    normals = ()
    for i in range(len(mesh)):
        normal = getNormal(mesh.vertices[mesh.polygons[i][0]],
                           mesh.vertices[mesh.polygons[i][1]],
                           mesh.vertices[mesh.polygons[i][2]])
        normals += (normal,)
    return normals


def gen_buffer(value, w, h):
    """ This function generates a buffer object. """
    line = []
    buffer = []
    for i in range(w):
        line.append(value)

    for i in range(h):
        buffer.append(copy.deepcopy(line))

    return buffer


def warn(key, exception):
    print("Warning : ", end="")
    if key == 0:
        print("File not found,", exception, "does not exist.")
    elif key == 1:
        print("Index out of range! ", exception, "Is not a valid index.")


def project(mesh, scale, h, v):
    """ This function handles perspective projection of points."""
    projected = ()

    # Project vertices
    for p in mesh.vertices:
        try:
            inv_z = -1 / p.z
            projected += (Vect3(((p.x * inv_z) * scale + h), ((p.y * inv_z) * scale + v), inv_z),)
        except ZeroDivisionError:
            projected += (p,)

    mesh.vertices = projected

    return mesh


def QuickSort(sort, index):
    """my implementation of the QuickSort algorithm originally writen by Tony Hoare, 1960."""

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
    sort[0], sort[current_position] = sort[current_position], sort[0]
    index[0], index[current_position] = index[current_position], index[0]

    # recursively sort blocks
    left = QuickSort(sort[0:current_position], index[0:current_position])
    right = QuickSort(sort[current_position + 1:elements], index[current_position + 1:elements])

    # recombine lists into one list
    return sort, left[1] + [index[current_position]] + right[1]


def lerp_factor(a, b, c):
    """ This function returns the amount of interpolation is required between a to b to end at c"""
    return (c - a) / ((c - a) - (c - b))


def interp(x1, y1, x2, y2, y):
    """ This function interpolates between two points at a given y."""
    try:
        result = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
        return result
    except ZeroDivisionError:
        return x1


def clipPolygonPlane(a, b, c, plane):
    """ This function handles clipping a polygon against a single plane. """
    a_distance, b_distance, c_distance = plane.pointToPlane(a.p), plane.pointToPlane(b.p), plane.pointToPlane(c.p)
    a_inside, b_inside, c_inside = a_distance > 0.001, b_distance > 0.001, c_distance > 0.001,
    inside = int(a_inside) + int(b_inside) + int(c_inside)

    if inside == 0:
        return None, None, 0

    elif inside == 1:
        if a_inside:
            b, c = plane.vertexPlaneIntersect(a, b), plane.vertexPlaneIntersect(a, c)
        elif b_inside:
            a, c = plane.vertexPlaneIntersect(b, a), plane.vertexPlaneIntersect(b, c)
        elif c_inside:
            b, a = plane.vertexPlaneIntersect(c, b), plane.vertexPlaneIntersect(c, a)
        return (a, b, c), None, 1

    elif inside == 2:
        if not a_inside:  # A is off-screen
            ab, ac = plane.vertexPlaneIntersect(a, b), plane.vertexPlaneIntersect(a, c)
            return (b, ab, ac), (c, b, ac), 2

        elif not b_inside:  # B is off-screen
            bc, ba = plane.vertexPlaneIntersect(b, c), plane.vertexPlaneIntersect(b, a)
            return (a, ba, bc), (a, c, bc), 2

        elif not c_inside:  # C is off-screen
            cb, ca = plane.vertexPlaneIntersect(c, b), plane.vertexPlaneIntersect(c, a)
            return (b, cb, ca), (b, a, ca), 2

    return None, None, 3
