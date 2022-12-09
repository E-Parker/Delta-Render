# This program was writen by Ethan Parker.
# This program is a self-contained 3d render engine.

import pygame
from time import perf_counter
from math import *
from constants import *
from render_math import *
from render_polygon import *
from collision import *
from pygame import PixelArray as PxArry
from pygame import Vector3 as Vect3
from pygame import Vector2 as Vect2
from copy import deepcopy as dc
    

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# CAMERA CLASS


class Camera:
    """ This class is a generic camera class used for rendering. """

    def __init__(self, near, far, fov, rWidth, rHeight, width, height, moveSpeed, coliderSize):
        # Display & rendering pre-calculations:
        self.resolution = (rWidth, rHeight)
        self.h_screen_center = width // 2   # Horizontal center for WindowSurface.
        self.v_screen_center = height // 2  # Vertical center for WindowSurface.
        self.fov = fov
        self.aspect_ratio = rWidth / rHeight
        self.scale = 1 / (tan(0.5 * radians(self.fov)))
        self.h_scale_const = 0.5 * (rWidth * self.scale / min((1, self.aspect_ratio)))
        self.v_scale_const = 0.5 * (rHeight * self.scale * max((1, self.aspect_ratio)))
        self.h_center = 0.5 * rWidth        # Horizontal center for screen surface. (may be different resolution).
        self.v_center = 0.5 * rHeight       # Vertical center for screen surface.    same goes for this. ^^^
        self.deviation = self.fov / 180     # used to check if objects can be seen by the camera.

        # Camera Clipping:
        self.near = near    # used to define near clip plane.
        self.far = far      # used to define far clip plane.
        self.maxDist = far * far    # check if an object is within view, no point in clipping something you can't see!
        self.clip = self.genClip()  # Generate planes from peramiters set above.

        # Boolens & States:
        self.active = True  # Bool for game pause / exit condition
        self.move = [False, False, False, False, False, False]  # bools for if camera is moving and in which direction.
        
        # Rotation:
        self.rotation_max = 85      # Maximum vertical rotation in degrees 
        self.x_rotation = 0
        self.y_rotation = 0
        self.z_rotation = 0

        # Movement:
        self.gravity = Gravity()
        self.mSpeed = moveSpeed               # movement speed multiplier
        self.maxSpeed = self.mSpeed * 2       # Max movement speed
        self.maxAirSpeed = self.maxSpeed * 2  # Max movement speed in air
        self.aceeleration = 0.40              # Amount of control over change in velocity
        self.airAceeleration = 0.10           # Amount of control while airborne
        self.radius = coliderSize             # colider radius
        self.height = Vect3(0,0.25,0)         # Height offset
        self.airborne = True                  # Bool for if camera is in the air

        # Vectors:
        self.d_rotation = Vect3(0, 0, 1)      # Vector for checking if a face is on screen, (rotation vector scaled by aspect ratio)
        self.position = Vect3(0, 0, 0)        # Position of the camera.
        self.rotation = Vect3(0, 0, 1)        # Unit vector in the direction the camera is facing.
        self.velocity = Vect3(0, 0, 0)        # Velocity of the camera.

        # Debug:
        self.noclip = True
        self.allow_movement = False

    def genClip(self):
        """ This method handles generating the points used to form the clipping planes. This can be precalculated with
            some basic trig. Basically, this method generates a thrustum mesh which can then be used to generate a Plane
            object. """

        # Solve for top and left sides of the far plane:
        horizontal_far_side_length = ((0 - self.h_center) * self.far) / self.h_scale_const
        vertical_far_side_length = ((0 - self.v_center) * self.far) / self.v_scale_const

        # Solve for top and left sides of the near plane:
        horizontal_near_side_length = ((0 - self.h_center) * self.near) / self.h_scale_const
        vertical_near_side_length = ((0 - self.v_center) * self.near) / self.v_scale_const

        # generate points:
        c1 = Vect3(horizontal_far_side_length, vertical_far_side_length, self.far)
        c2 = Vect3(horizontal_far_side_length, -vertical_far_side_length, self.far)
        c3 = Vect3(horizontal_near_side_length, vertical_near_side_length, self.near)
        c4 = Vect3(horizontal_near_side_length, 0 - vertical_near_side_length, self.near)
        c5 = Vect3(0 - horizontal_far_side_length, vertical_far_side_length, self.far)
        c6 = Vect3(0 - horizontal_far_side_length, 0 - vertical_far_side_length, self.far)
        c7 = Vect3(0 - horizontal_near_side_length, vertical_near_side_length, self.near)
        c8 = Vect3(0 - horizontal_near_side_length, 0 - vertical_near_side_length, self.near)

        # Generate planes from points.
        ner = Plane(c7, c4, c3)  # Near 
        far = Plane(c6, c5, c1)  # Far 
        top = Plane(c6, c2, c4)  # Top 
        btm = Plane(c1, c5, c7)  # Bottom 
        lft = Plane(c3, c2, c1)  # Left 
        rgt = Plane(c5, c6, c8)  # right 

        # planes stored in this order so the game has the least amount of work to do per clip operation.
        planes = (ner, lft, rgt, top, btm, far)

        return planes

    def cameraClipMesh(self, mesh):
        """ This method clips a mesh against the camera's clip plane. It first removes any back-facing faces, 
            Then, cookie-cutters out a section of the mesh that when projected from world-space to screen-space, 
            will perfectly fit the screen size. This is done to avoid drawing infinately large polygons 
            (as x,y -> 0, x,y out -> inf) and to avoid a divide by zero error if x,y = 0. """
        
        # Remove back-facing Faces:
        index = 0
        while index < len(mesh):
            if mesh[index][3].dot(self.d_rotation) > self.deviation: 
                del mesh[index]
            else: 
                index += 1
        
        # Clip Mesh:
        clipped = clipMesh(self.r_clip, mesh)

        # Move and rotate cliped section such that it fits 
        clipped.move(-self.position)
        clipped.rotate_y(self.y_rotation)
        clipped.rotate_x(self.x_rotation)
        return clipped

    def update(self, frameDelta, colliders):
        """ This function handles updating the camera and it's properties. """

        # offset position by height:
        self.position -= self.height

        # calculate friction depending on if the camera is in the air or on the ground
        if self.airborne and not self.noclip:
            self.velocity = self.velocity * 0.98
        else:
            self.velocity = self.velocity * 0.70
        
        vel = Vect3(0, 0, 0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.active = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not self.airborne:
                        self.gravity.set_jumping()
                        self.airborne = True

                if event.key == pygame.K_ESCAPE:
                    self.active = False

                # Check for keyboard input:
                if event.key == ord('k'): self.allow_movement = not self.allow_movement
                if event.key == ord('t'): self.noclip = not self.noclip
                if event.key == ord('w'): self.move[0], self.move[1] = True, False
                if event.key == ord('s'): self.move[1], self.move[0] = True, False
                if event.key == ord('a'): self.move[2], self.move[3] = True, False
                if event.key == ord('d'): self.move[3], self.move[2] = True, False
                if event.key == ord('r'): self.move[4], self.move[5] = True, False
                if event.key == ord('f'): self.move[5], self.move[4] = True, False

            elif event.type == pygame.KEYUP:
                if event.key == ord('w'): self.move[0] = False
                if event.key == ord('s'): self.move[1] = False
                if event.key == ord('a'): self.move[2] = False
                if event.key == ord('d'): self.move[3] = False
                if event.key == ord('r'): self.move[4] = False
                if event.key == ord('f'): self.move[5] = False
        
        for test in self.move:
            if test:
                if self.move[0]: vel.z += self.mSpeed
                if self.move[1]: vel.z -= self.mSpeed
                if self.move[2]: vel.x += self.mSpeed
                if self.move[3]: vel.x -= self.mSpeed
                if self.noclip:  # when player is in noclip, alow for up and down movement
                    if self.move[4]: vel.y += self.mSpeed
                    if self.move[5]: vel.y -= self.mSpeed
        
        if vel.length_squared() != 0:
            vel.scale_to_length(self.mSpeed)
            vel.rotate_y_ip(self.y_rotation)
        
        # Check if character is in the air:
        if self.airborne and not self.noclip:
            self.velocity += vel * self.airAceeleration
            if self.velocity.length_squared() > self.maxAirSpeed * self.maxAirSpeed:
                self.velocity.scale_to_length(self.maxAirSpeed)
        else:
            self.velocity += vel * self.aceeleration
            if self.velocity.length_squared() > self.maxSpeed * self.maxSpeed:
                self.velocity.scale_to_length(self.maxSpeed)
        
        if self.noclip:
            self.position += self.velocity * frameDelta

        else:
            # Calculate Gravity:
            self.gravity.update(frameDelta)
            vel = (self.velocity + Vect3(0, 0.05 * self.gravity.current_value, 0)) * frameDelta
            
            # Check for Collisions:
            collisions = 0
            floors = 0
            for collider in colliders:
                if collider.enabled:
                    collision = collider.CollideMesh(self.position, vel, self.radius)
                    self.position = collision[0]
                    collisions += collision[1]
                    floors += collision[2]
            if collisions > 0:
                self.gravity.set_falling()  # if not coliding, character is falling.

            self.airborne = floors == 0

        # Get mouse position
        mouse_position = pygame.mouse.get_pos()
        pygame.mouse.set_pos((self.h_screen_center, self.v_screen_center))

        # Update Vectors:
        y_rotation = degrees((mouse_position[0] - self.h_screen_center) * 0.0025)
        x_rotation = degrees((mouse_position[1] - self.v_screen_center) * 0.0025)
        
        if self.x_rotation - x_rotation >= self.rotation_max: x_rotation = self.x_rotation - self.rotation_max
        elif self.x_rotation - x_rotation <= -self.rotation_max: x_rotation = self.x_rotation + self.rotation_max
        
        if self.allow_movement:
            self.y_rotation -= y_rotation
            self.x_rotation -= x_rotation

            self.rotation = Vect3(0, 0, 1)
            self.rotation = self.rotation.rotate_x(-self.x_rotation)
            self.rotation = self.rotation.rotate_y(self.y_rotation)
        else:
            self.position = Vect3(0, 1, 0)

        self.d_rotation = Vect3(self.rotation)
        self.d_rotation.y *= self.aspect_ratio
        
        self.position += self.height

        # Update clip planes to camera rotation:
        # This avoids rotating and translating every vertex in every mesh.
        # only the vertecies left on screen are actually moved.

        self.r_clip = dc(self.clip)
        for plane in self.r_clip:
            plane.rotate_x(-self.x_rotation)
            plane.rotate_y(-self.y_rotation)
            plane.p += self.position
            plane.update()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# DRAWING FUNCTIONS


def display_text(windowSurface, line, text, font, linetype='l', fg=(40, 40, 40), transparent=False):
    """ This function writes text on a pygame.Surface object. """
    
    # Variables:
    # windowSurface     -     Surface the text is drawn on
    # line              -     what line of text to display on
    # text              -     Text to be displayed

    # set up the text
    if transparent:
        text = font.render(text, False, fg)
    else:
        text = font.render(text, False, fg, (255, 255, 255))
    textRect = text.get_rect()

    textRect.top = windowSurface.get_rect().top + (font.get_height() * line)
    if linetype == 'l':    # center to left side
        textRect.left = windowSurface.get_rect().left
    elif linetype == 'r':  # center to right side
        textRect.right = windowSurface.get_rect().right
    elif linetype == 'c':  # center text
        textRect.centerx = windowSurface.get_rect().centerx

    # draw the text's background rectangle onto the surface
    windowSurface.blit(text, textRect)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# GAME CLASS AND MAINLINE


class Game:
    def __init__(self, fov=FOV, far=FARCLIP, near=NEARCLIP, rWidth=RWIDTH, rHeight=RHEIGHT, width=WIDTH, height=HEIGHT, moveSpeed=MSPEED, colliderSize=COLIDER_SIZE, skytex=SKYTEX):
        """ This class contains most of the methods used to run the program. I used a class here to allow for easier instanceing. """
        
        self.meshes = []        # List of mesh objects
        self.textures = []      # List of textures used by said mesh objects
        self.colliders = []     # List of colliders

        # Loading game assets:
        self.font = pygame.font.SysFont("courier new", 12)                           # it's a font... who would'a guessed??
        self.missingTexture = pygame.image.load(DD_TEXTR + "missing.png").convert()  # if an image fails to load, use this.
        skytex = pygame.image.load(DD_TEXTR + skytex).convert()                      # Skybox Texture
        skytex = pygame.transform.flip(skytex, True, False)                          # Because of some weird inverted rendering jank, the texture is fliped.
        self.skytex = (skytex, skytex.get_width() - 1, skytex.get_height() - 1)
        self.skybox = UnpackMesh(DD_ASSET + SKYBOX)                                  # Load Skybox mesh

        # Rendering:
        self.frameDelta = 1
        self.camera = Camera(near, far, fov, rWidth, rHeight, width, height, moveSpeed, colliderSize)
        self.screen = pygame.Surface((rWidth, rHeight)).convert()   # Set up local screen surface, scaled to actual window after rendering.
        self.pixelArray = PxArry(self.screen)                       # Pixel array is used for polygon drawing functions because it's faster than the .setpixel() method.
        self.screen.lock()                                          # Since writes will only happen dirrectly to the pixel array, the screen can be locked to improve performance.
        self.depth_buffer = ()
        self.initiateBuffer()

        #Debug:
        self.chunksize = 4
        self.runtime = perf_counter()
        self.logicTime = 0
        self.renderTime = 0

    def loadFiles(self, meshes, colliders, textures):
        """ This method loads meshes and textures into the game. """
        # Load Textures:
        for filename in textures:
            try:
                texture = pygame.image.load(TEXTR + filename).convert()
                texture = pygame.transform.flip(texture, True, False)       # I goofed up and inverted all the math, its easier to just flip it here.
            except FileNotFoundError:
                texture = self.missingTexture
            
            # Store textures with their size for quick access. Avoids getatrr() call with .getwidth() & .getheight() methods.
            texture = (texture, texture.get_width() - 1, texture.get_height() - 1)
            self.textures.append(texture)

        # Load Meshes:
        for i in range(len(meshes)):
            mesh = UnpackMesh(ASSET + meshes[i][0])
            mesh.texIndex = meshes[i][1]
            mesh.static = meshes[i][2]
            self.meshes.append(mesh)

        # Load colliders:
        for filename in colliders:
            self.colliders.append(MeshCollider(UnpackMesh(CLLDR + filename)))

    def commitGameMeshes(self, size):
        """ This method, when called, fragments all static meshes in a scene. 
            WARNING, STATIC MOVE METHOD WILL BREAK AFTER THIS IS CALLED. ONLY USE ON BACKGROUND/LEVEL GEOMETRY. """
        
        fragmented = []  # List to store fragmented meshes.
        
        index = 0
        while index < len(self.meshes):
            mesh = self.meshes[index]
            if mesh.static:
                # if the mesh is static, fragment it and append it to the fragmented list.
                fragmented.extend(fragMesh(mesh, size))
                del self.meshes[index]
            else:
                index += 1
        
        # adding all the fragmented meshes back to this list,
        # dynamic meshes can still be indexed properly.
        self.meshes.extend(fragmented)

    def initiateBuffer(self):
        """ This function clears the depth buffer. """
        # for every horizontal line of the buffer:
        dLine = [self.camera.far for _ in range(self.camera.resolution[0])]
        
        # for every vertical line of the buffer:
        self.depth_buffer = tuple([dc(dLine) for _ in range(self.camera.resolution[1])])

    def run_logic(self, clock):
        
        # Set Perfomance counter:
        t1 = perf_counter()

        # Get Frame Delta:
        self.frameDelta = clock.tick(60) * 0.001 * FPS
        
        # Update Camera:
        self.camera.update(self.frameDelta, self.colliders)

        if not self.camera.active:
            runtime = self.runtime - perf_counter()
            print("logic time:\t", self.logicTime / runtime)
            print("render Time:\t", self.renderTime / runtime)
            terminate()

        # Update performance Counter:
        self.logicTime += t1 - perf_counter()
    
    def projectStatic(self, points):
        hs, vs, hc, vc = self.camera.h_scale_const, self.camera.v_scale_const, self.camera.h_center, self.camera.v_center
        for p in points.vertices:
            inv_z = -1 / p.z
            p.update(((p.x * inv_z) * hs) + hc, ((p.y * inv_z) * vs) + vc, inv_z)
        return points

    def checkMesh(self, origin, radius):
        """ This method checks a mesh against the camera's planes to check if it can be seen or not. """
        point = origin - self.camera.position
        if point.length_squared() > self.camera.maxDist:
            return False
        point.rotate_y_ip(-self.camera.y_rotation)
        point.rotate_x_ip(self.camera.x_rotation)
        for plane in self.camera.clip:
            if plane.pointToPlane(point) < -radius:
                return False
        return True

    def drawSkybox(self):
        """ This method handles drawing the sky box on top of the rendered screen. """
        # Create a copy to preserve the original mesh.
        skybox = dc(self.skybox)                

        # Rotate sky box such that it appears fixed in rotation:
        skybox.rotate_y(self.camera.y_rotation)
        skybox.rotate_x(self.camera.x_rotation)

        # Clip S\sky box to view:
        skybox = clipMesh(self.camera.clip,skybox)
        skybox = self.projectStatic(skybox)
        RasterizeSkybox(self.pixelArray, self.screen, skybox, self.skytex, TRANSPARENCY)

    def render(self, Display):
        """ This function renders the scene to a surface. """
        t1 = perf_counter()
        projected = []
        pos = self.camera.position

        # Clear screen / buffer(s):
        depth_buffer = dc(self.depth_buffer)

        if RENDERSKY:
            self.pixelArray[:] = (255, 0, 255)

        # Render mesh(s):
        for mesh in self.meshes:
            if self.checkMesh(mesh.Origin(), mesh.farPoint):
                meshCopy = dc(mesh)
                if not meshCopy.static:
                    if meshCopy.rotation != [0,0,0]: 
                        meshCopy.rotate(meshCopy.rotation)
                        meshCopy.rotateNormals(meshCopy.rotation)
                        meshCopy.updateBrightness()
                    if meshCopy.position[:] != [0,0,0]:
                        meshCopy.move(meshCopy.position)
                
                meshCopy = self.camera.cameraClipMesh(meshCopy)
                meshCopy.getDistance()
                projected.append(self.projectStatic(meshCopy))

        # Sort meshes by distance to camera, draw the nearest ones first to avoid overdrawing.
        distances = [(mesh.origin + pos).length_squared() for mesh in projected]
        distances = QuickSort(distances, [i for i in range(len(projected))])[1]
        distances.reverse()

        if len(projected) != 0:
            texIndex = 0                 # Default to texture No.0
            t = self.textures[texIndex]  # Get texture data
            # Draw remaining meshe(s) / polygons:  
            for i in distances:
                mesh = projected[i]
                if mesh.texIndex != texIndex: texIndex, t = mesh.texIndex, self.textures[mesh.texIndex]
                # QuickSort is memory intensive and slow (in Python), only run Quicksort if length is < 500.
                if len(mesh) < SORTLIMIT: indecies = QuickSort(mesh.depth, [i for i in range(len(mesh))])[1] 
                else:                     indecies = range(len(mesh)) 
                RasterizeMesh(self.pixelArray, depth_buffer, mesh, t, indecies)

        # Draw sky box:
        if RENDERSKY and TRANSPARENCY in self.pixelArray: 
            self.drawSkybox()

        # Scale to screen size:
        scaled = pygame.transform.scale(self.screen, (WIDTH, HEIGHT))
        Display.blit(scaled, (0, 0))
        
        # Draw debug text:
        display_text(Display, 0, "FPS | " + str(round((60 / self.frameDelta))), self.font)
        pygame.display.flip()

        # Update performance Counter
        self.renderTime += t1 - perf_counter()
    

def main():
    pygame.init()
    pygame.mouse.set_visible(False)
    
    clock = pygame.time.Clock()
    flags = pygame.DOUBLEBUF# | pygame.FULLSCREEN
    Display = pygame.display.set_mode((WIDTH, HEIGHT), flags)
    icon = pygame.image.load("dependencies/textures/icon.png").convert_alpha()

    pygame.display.set_caption("Software Render Version 7 - 0.0.9")
    pygame.display.set_icon(icon)

    game = Game()

    props = (('ship.obj', 0, True),)
    textures = ('ship.png',)
    colliders = ('ship.obj',)
    
    game.loadFiles(props, colliders, textures)

    while 1:
        game.run_logic(clock)
        game.render(Display)

main()

