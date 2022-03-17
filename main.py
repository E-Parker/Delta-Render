# This program is the mainline for version 2.8 of ΔRender
# this program was writen by Ethan Parker


import func
import make
import render
import pygame
import math
import copy
from pygame import PixelArray as PxArry
from pygame.math import Vector3 as Vect3
from pygame.math import Vector2 as Vect2

SCALE = 6
FOV = 90
SCREENSIZE = (1024, 576)


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

        # Initialize Camera:
        self.render_resolution = (screensize[0] // res, screensize[1] // res)
        self.camera = func.Camera(0.1, 32, fov, self.render_resolution, screensize)
        self.scale = res

        # Display Settings:
        self.font = pygame.font.SysFont("courier new", 12)
        self.screen = pygame.Surface(self.render_resolution).convert()
        self.screen.set_colorkey(render.TRANSPARENCY)
        self.screen.lock()
        self.pixel_array = PxArry(self.screen)

        # Render settings:
        self.missing_texture = pygame.image.load("dependencies/textures/missing.png").convert()
        self.sky_colour = pygame.Color(250, 250, 240)

        # Buffer Objects:
        self.depth = func.gen_buffer(self.camera.far, self.render_resolution[0], self.render_resolution[1])

        # Vectors:
        self.light_vector = Vect3(0, -0.75, -0.25)
        self.origen = Vect3(0, 0, 0)

        # Booleans:
        self.is_active = True

        # Debug:
        self.onscreen = 0

    def clipMesh(self, original_mesh, textures_index):
        """ This function handles clipping the scene. """
        mesh = make.copy_Mesh(original_mesh)
        # preliminary pass to get rid of back facing polygons:
        index = 0
        while index < len(mesh):
            face = mesh[index]
            n = func.getNormal(face.a, face.b, face.c)
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
                a, b, c = (face.a, face.u), (face.b, face.v), (face.c, face.w)
                a_distance, b_distance, c_distance = plane.pointToPlane(a[0]), plane.pointToPlane(b[0]), plane.pointToPlane(c[0])
                a_inside, b_inside, c_inside = a_distance > 0.001, b_distance > 0.001, c_distance > 0.001,
                inside = int(a_inside) + int(b_inside) + int(c_inside)

                if inside == 0:  # Face is off-screen, don't render. (0 faces)
                    del mesh[index]
                    del textures_index[index]
                else:

                    if inside == 1:  # Two points off-screen, clip into trigon, update face.
                        if a_inside:
                            b, c = plane.vertexPlaneIntersect(a, b), plane.vertexPlaneIntersect(a, c)
                        elif b_inside:
                            a, c = plane.vertexPlaneIntersect(b, a), plane.vertexPlaneIntersect(b, c)
                        elif c_inside:
                            b, a = plane.vertexPlaneIntersect(c, b), plane.vertexPlaneIntersect(c, a)
                        face.update(a, b, c)
                        mesh[index] = face

                    elif inside == 2:  # One point off-screen. Clip into quad then trigon, update and append face.
                        new_trigon = func.Trigon()
                        if not a_inside:  # A is off-screen
                            ab, ac = plane.vertexPlaneIntersect(a, b), plane.vertexPlaneIntersect(a, c)
                            new_trigon.update(b, ab, ac)
                            face.update(c, b, ac)

                        elif not b_inside:  # B is off-screen
                            bc, ba = plane.vertexPlaneIntersect(b, c), plane.vertexPlaneIntersect(b, a)
                            new_trigon.update(a, ba, bc)
                            face.update(a, c, bc)

                        elif not c_inside:  # C is off-screen
                            cb, ca = plane.vertexPlaneIntersect(c, b), plane.vertexPlaneIntersect(c, a)
                            new_trigon.update(b, cb, ca)
                            face.update(b, a, ca)
                        mesh[index] = face
                        mesh.append(new_trigon)
                        textures_index.append(textures_index[index])

                    index += 1

        return mesh, textures_index

    def clear_depth(self):
        """ This function clears the depth buffer. """
        # for all every line of the buffer:
        for y in range(self.render_resolution[1]):
            self.depth[y] = [self.camera.far for x in range(self.render_resolution[0])]

    def draw_skybox(self):
        """ This function handles drawing the skybox after the rest of the scene has been drawn. """
        skybox = make.copy_Mesh(self.skybox)
        skybox.rotate_y(self.camera.y_rotation)
        skybox.rotate_x(-self.camera.x_rotation)
        skybox = self.clipMesh(skybox, list(range(len(skybox))))[0]
        projected = func.project(skybox, self.camera)
        render.renderSkybox(projected, self.screen, self.pixel_array, self.skybox_texture)

    def render(self):
        """ This function handles rendering the whole scene to self.screen for display. """
        # Update scene:
        new_mesh = make.Combine_Mesh(self.meshes)
        self.world, self.textures, self.textures_index = new_mesh[0], new_mesh[1], new_mesh[2]
        world_position = self.origen - self.camera.position - Vect3(0, self.camera.height, 0)
        self.world.move(world_position)
        self.world.rotate_y(self.camera.y_rotation)
        self.world.rotate_x(-1 * self.camera.x_rotation)

        # Clip, Project and rasterize points:
        clipped = self.clipMesh(self.world, self.textures_index)
        self.world, self.textures_index = clipped[0], clipped[1]

        # Clear buffers from last frame:
        self.pixel_array[:] = render.TRANSPARENCY  # self.sky_colour
        self.clear_depth()
        self.world.update()
        self.onscreen = len(self.world)
        # Render Scene:
        projected = func.project(self.world, self.camera)
        render.render(projected, self.pixel_array, self.depth, self.textures, self.textures_index, self.camera.filtering_toggle)

    def run_logic(self, clock, fps):
        """ Handles all logic such as collision detection and frame delta. """
        # Get Frame Delta:
        self.frame_delta = clock.tick(60) * 0.001 * fps
        self.is_active = self.camera.update(self.colliders, self.frame_delta)

    def draw_screen(self, window):
        self.render()
        self.draw_skybox()
        # Update Display:
        screen_scale = pygame.transform.scale(self.screen, window.get_size())
        window.blit(screen_scale, (0, 0))
        # func.display_text(window, 0, "O_S|" + str(self.onscreen), self.font, (0, 0, 0), 'c')
        func.display_text(window, 0, "FPS|" + str(round(60 / self.frame_delta)), self.font, (0, 0, 0), 'c')
        pygame.display.flip()


def main():
    pygame.init()

    fps = 60
    clock = pygame.time.Clock()

    # DISPLAY
    window = pygame.display.set_mode(SCREENSIZE)
    pygame.display.set_caption("Software Render")

    meshes = (make.Mesh("assets/meshes/ship.obj", "assets/textures/ship.png"),
              # make.Mesh("assets/meshes/water.obj", "assets/textures/water.png"),
              )

    colliders = (make.Collider("assets/colliders/ship.obj", 0.25, True),
                 )

    game = Game(meshes, colliders, SCREENSIZE, SCALE, FOV)
    pygame.mouse.set_visible(False)

    # i = 0
    while True:
        # i += 0.01 * game.frame_delta
        # game.meshes[1].move_to(Vect3(0, -3, i % 32))
        game.run_logic(clock, fps)
        game.draw_screen(window)
        if not game.is_active:
            break


main()
