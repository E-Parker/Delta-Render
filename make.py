# This program handles asset creation / reassignment for version 2.4 of ΔRender
# This program was writen by Ethan Parker

import func
import pygame
from pygame.math import Vector2 as Vect2
from pygame.math import Vector3 as Vect3


def Collider(filename, gap, enabled):
    """ This function handles loading of mesh colliders."""
    new_mesh = func.Mesh()
    obj = open(filename, "r")

    try:  # Try loading mesh
        for line in obj:
            line = line.strip("\n")
            line = line.split(" ")

            if line[0] == 'v':  # Vertex
                point = Vect3(float(line[1]), float(line[2]), float(line[3]))
                new_mesh.vertices += (point,)

            if line[0] == 'vt':  # UV texture information
                new_mesh.uv_vertices += (Vect2(0, 0),)

            elif line[0] == 'f':  # Polygon
                face1, face2, face3 = line[1].split('/'), line[2].split('/'), line[3].split('/')
                polygon = (int(face1[0]) - 1, int(face2[0]) - 1, int(face3[0]) - 1,)
                uv_polygon = (int(face1[1]) - 1, int(face2[1]) - 1, int(face3[1]) - 1,)
                new_mesh.polygons += (polygon,)
                new_mesh.uv_polygons += (uv_polygon,)
        # Calls update to write depth information.
        new_mesh.update()
    finally:
        obj.close()

    new_collider = func.Collider(new_mesh, gap, enabled)

    return new_collider


def Mesh(filename, texture):
    new_mesh = func.Mesh()

    try:
        obj = open(filename, "r")
    except FileNotFoundError:  # if mesh not found, load fallback.
        func.warn(0, filename)
        obj = open("dependencies/meshes/fallback.obj", "r")
        texture = "dependencies/textures/missing_2.png"

    try:  # Try to load specified Texture
        loaded_texture = pygame.image.load(texture).convert()
    except FileNotFoundError:  # File not found, replace with missing texture
        func.warn(0, texture)
        loaded_texture = pygame.image.load("dependencies/textures/missing.png").convert()

    new_mesh.texture = pygame.transform.flip(loaded_texture, False, True)

    try:  # Try loading mesh
        for line in obj:
            line = line.strip("\n")
            line = line.split(" ")

            if line[0] == 'v':  # Vertex
                point = Vect3(float(line[1]), float(line[2]), float(line[3]))
                new_mesh.vertices += (point,)

            if line[0] == 'vt':  # UV texture information
                uv = Vect2(float(line[1]), float(line[2]))
                new_mesh.uv_vertices += (uv,)

            elif line[0] == 'f':  # Polygon
                face1, face2, face3 = line[1].split('/'), line[2].split('/'), line[3].split('/')
                polygon = (int(face1[0]) - 1, int(face2[0]) - 1, int(face3[0]) - 1,)
                uv_polygon = (int(face1[1]) - 1, int(face2[1]) - 1, int(face3[1]) - 1,)
                new_mesh.polygons += (polygon,)
                new_mesh.uv_polygons += (uv_polygon,)
        # Calls update to write depth information.
        new_mesh.update()
    finally:
        obj.close()

    return new_mesh


def Combine_Mesh(meshes):
    """ This function combines n number of meshes into one mesh."""

    # Create new mesh to add information to
    new_mesh = func.Mesh()
    textures = []
    textures_index = []

    for x in range(len(meshes)):
        current_mesh = meshes[x]
        textures.append(current_mesh.texture)
        index_offset = len(new_mesh.vertices)
        tex_index_offset = len(new_mesh.uv_vertices)

        new_mesh.vertices += current_mesh.vertices
        new_mesh.uv_vertices += current_mesh.uv_vertices

        # Offset polygons by the new index:
        for index in range(len(current_mesh.polygons)):
            textures_index.append(x)
            new_mesh.polygons += ((current_mesh.polygons[index][0] + index_offset,
                                   current_mesh.polygons[index][1] + index_offset,
                                   current_mesh.polygons[index][2] + index_offset),)

        # Offset texture polygons the new index:

        for index in range(len(current_mesh.uv_polygons)):
            new_mesh.uv_polygons += ((current_mesh.uv_polygons[index][0] + tex_index_offset,
                                      current_mesh.uv_polygons[index][1] + tex_index_offset,
                                      current_mesh.uv_polygons[index][2] + tex_index_offset),)

    return new_mesh, textures, textures_index


def copy_Mesh(mesh):
    new_mesh = func.Mesh()
    new_mesh.vertices = mesh.vertices
    new_mesh.uv_vertices = mesh.uv_vertices
    new_mesh.polygons = mesh.polygons
    new_mesh.uv_polygons =mesh.uv_polygons
    return new_mesh

