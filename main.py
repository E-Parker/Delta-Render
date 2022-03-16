# This program is the mainline for version 2.4 of ΔRender
# this program was writen by Ethan Parker


# import copy
import func
import make
import pygame
import math
import render
from pygame.locals import *
from pygame.math import Vector3 as Vect3
from pygame.math import Vector2 as Vect2

SCALE = 7
RENDER_SCALE = 500
SCREENSIZE = (1280, 720)

class Game:
    def __init__(self, meshes, colliders, screensize, res, fov):

        # Initialize Objects:
        self.skybox = make.Mesh("dependencies/meshes/skybox.obj", "dependencies/textures/skybox_default.png")
        self.skybox_texture = self.skybox.texture
        self.meshes = meshes
        self.colliders = colliders
        self.world = func.Mesh()
        self.textures = []
        self.textures_index = []
        self.frame_delta = 0
        self.is_active = True

        # Display Settings:
        self.screensize = screensize
        self.screensize_center = (screensize[0] / 2, screensize[1] / 2)
        self.scale = res
        self.render_scale = fov / self.scale
        self.render_size = (screensize[0] // res, screensize[1] // res)
        self.center = (self.render_size[0] // 2, self.render_size[1] // 2,)
        self.screen = pygame.Surface(self.render_size).convert()
        self.screen.lock()
        print("render scale",self.render_scale)

        # Render settings:
        self.camera = func.Camera(fov, 0.25, 16, self.render_size, self.render_scale)
        self.missing_texture = pygame.image.load("dependencies/textures/missing.png").convert()
        self.sky_colour = pygame.Color(250, 250, 240)

        # Buffer Objects:
        self.depth = func.gen_buffer(self.camera.far, self.render_size[0], self.render_size[1])
        self.screen_write = pygame.mask.Mask(self.render_size)

        # Vectors:
        self.light_vector = Vect3(0, -0.25, -0.75)
        self.origen = Vect3(0, 0, 0)

    def clipMesh(self, mesh, textures_index):
        """ This function handles clipping the scene. """

        # preliminary pass to get rid of back facing polygons:
        index = 0
        while index < len(mesh):
            face = mesh[index]
            n = func.getNormal(face.a.p, face.b.p, face.c.p)
            if n.dot(self.camera.rotation) > self.camera.deviation:
                del mesh[index]
                del textures_index[index]
            else:
                index += 1

        # clip mesh against all planes:
        for plane in self.camera.clip:
            index = 0
            while index < len(mesh):
                face = mesh[index]
                new_face = func.clipPolygonPlane(face.a, face.b, face.c, plane)

                if new_face[2] == 0:  # Face is off-screen, don't render. (0 faces)
                    del mesh[index]
                    del textures_index[index]
                    index -= 1

                elif new_face[2] == 1:  # Two points off-screen, clip into trigon. (1 face)
                    face.update(new_face[0][0], new_face[0][1], new_face[0][2])
                    mesh[index] = face

                elif new_face[2] == 2:  # one point is off-screen, clip into two trigons. (2 faces)
                    new_trigon = func.Trigon()
                    new_trigon.update(new_face[0][0], new_face[0][1], new_face[0][2])
                    face.update(new_face[1][0], new_face[1][1], new_face[1][2])
                    mesh[index] = face
                    mesh.append(new_trigon)
                    try:
                        textures_index.append(textures_index[index])
                    except IndexError:
                        pass
                index += 1

        return mesh, textures_index

    def clear_depth(self):
        """ This function clears the depth buffer. """

        # for all every line of the buffer:
        for y in range(self.render_size[1]):
            line = self.depth[y]
            # for every element of the line:
            for x in range(self.render_size[0]):
                if self.screen_write.get_at((x, y)) == 1:  # if the pixel has been set, reset it.
                    line[x] = self.camera.far

    def draw_skybox(self):
        """ This function handles drawing the skybox after the rest of the scene has been drawn. """
        skybox = make.copy_Mesh(self.skybox)
        skybox.rotate_y(self.camera.y_rotation)
        skybox.rotate_x(-self.camera.x_rotation)
        skybox = self.clipMesh(skybox, list(range(len(skybox))))[0]
        projected = func.project(skybox, self.camera.size, self.center[0], self.center[1])
        render.renderSkybox(projected, self.screen, self.screen_write, self.skybox_texture)

    def render(self):
        """ This function handles rendering the whole scene to self.screen for display. """
        # Update scene:
        new_mesh = make.Combine_Mesh(self.meshes)
        self.world, self.textures, self.textures_index = new_mesh[0], new_mesh[1], new_mesh[2]
        self.world.move(self.origen - self.camera.position)
        self.world.rotate_y(self.camera.y_rotation)
        self.world.rotate_x(-1 * self.camera.x_rotation)

        # Clip, Project and rasterize points:
        clipped = self.clipMesh(self.world, self.textures_index)
        self.world, self.textures_index = clipped[0], clipped[1]

        # Clear buffers from last frame:
        self.clear_depth()
        self.screen_write.clear()
        self.world.update()

        # Render Scene:
        projected = func.project(self.world, self.camera.size, self.center[0], self.center[1])
        render.render(projected, self.screen, self.depth, self.screen_write, self.textures, self.textures_index, self.camera.far)

    def run_logic(self, clock, fps):
        """ Handles all logic such as collision detection and frame delta. """
        # Get Frame Delta:
        self.frame_delta = clock.tick(60) * 0.001 * fps
        self.is_active = self.camera.update(self.colliders, self.frame_delta, self.screensize_center)

    def draw_screen(self, window):
        self.render()
        self.draw_skybox()

        # Update Display:
        screen_scale = pygame.transform.scale(self.screen, self.screensize)
        window.blit(screen_scale, (0, 0))
        pygame.display.flip()


def main():
    pygame.init()

    fps = 60
    clock = pygame.time.Clock()

    # DISPLAY
    window = pygame.display.set_mode(SCREENSIZE)
    pygame.display.set_caption("Software Render")
    # font = pygame.font.SysFont("courier new", 12)

    meshes = (make.Mesh("assets/meshes/cube.obj", "assets/textures/gradient.png"),
              make.Mesh("assets/meshes/plane.obj", "assets/textures/testing2.png"),
              )

    colliders = (make.Collider("assets/colliders/test_collider.obj", 0.25, True),
                 )

    game = Game(meshes, colliders, SCREENSIZE, SCALE, RENDER_SCALE)
    pygame.mouse.set_visible(False)
    pygame.mouse.set_pos(game.screensize_center)
    game.meshes[0].move_to(Vect3(0, -0.5, 0))
    game.meshes[1].move_to(Vect3(0, -1.25, 0))

    while True:
        game.run_logic(clock, fps)
        game.draw_screen(window)
        if not game.is_active:
            break


main()
