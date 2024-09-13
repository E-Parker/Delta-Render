# This program was written by Ethan Parker.
# This program is the mainline for my 3d render engine.

import pygame
from time import perf_counter
from math import *
from constants import *
from render_math import *
from render_polygon import *
from render_camera import *
from collision import *
from pygame import PixelArray as PxArry
from copy import deepcopy as dc


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
    def __init__(self, fov=FOV, far=FAR_CLIP, near=NEAR_CLIP, rWidth=R_WIDTH, rHeight=R_HEIGHT, width=WIDTH, height=HEIGHT, moveSpeed=M_SPEED, colliderSize=COLLIDER_SIZE, skytex=SKYTEX, fragment_size=FRAGMENT_SIZE):
        """ This class contains most of the methods used to run the program. I used a class here to allow for easier instancing. """

        print(R_WIDTH)
        
        self.meshes = []                # Dynamic mesh objects
        self.colliders = []             # Dynamic Mesh colliders
        self.fragmented_meshes = []     # Static meshes, faster culling
        self.fragmented_colliders = []  # Static mesh colliders
        self.textures = []              # Textures used by mesh objects
        
        self.r_meshes = []              # List of transformed meshes
        self.r_colliders = []           # List of transformed colliders

        self.culled_meshes = []         # meshes that are visible on any frame

        # Loading game assets:
        self.font = pygame.font.SysFont("courier new", 12)                               # it's a font... who would'a guessed??
        self.missingTexture = pygame.image.load(DEP_TEXTURES + "missing.png").convert()  # if an image fails to load, this is used instead.
        
        self.missingObject = UnpackMesh(DEP_ASSETS + MISSING_OBJECT)      
        missingObjTex = pygame.image.load(DEP_TEXTURES + "missing_object.png").convert()
        self.missingObjectTexture = (missingObjTex, missingObjTex.get_width() - 1, missingObjTex.get_height() - 1)

        skytex = pygame.image.load(DEP_TEXTURES + skytex).convert()                      # Skybox Texture
        self.skytex = (skytex, skytex.get_width() - 1, skytex.get_height() - 1)
        self.skybox = UnpackMesh(DEP_ASSETS + SKYBOX)                                    # Load Skybox mesh

        # Rendering:
        self.fragment_size = fragment_size                         # size of each chunk. 
        self.frameDelta = 0
        self.camera = Camera(near, far, fov, rWidth, rHeight, width, height, moveSpeed, colliderSize)
        self.screen = pygame.Surface((rWidth, rHeight)).convert()  # Set up local screen surface, scaled to actual window after rendering.
        self.pixelArray = PxArry(self.screen)                      # Pixel array is used for polygon drawing functions because it's faster than the .setpixel() method.
        self.screen.lock()                                         # Since writes will only happen directly to the pixel array, the screen can be locked to improve performance.
        self.depth_buffer = ()
        self.initiateBuffer()

        # Loading Screen:
        self.loading_background = pygame.image.load(DEP_TEXTURES + "loading_background.png").convert()
        self.loading_background = pygame.transform.smoothscale(self.loading_background, (WIDTH, HEIGHT))
        self.text_lines = ["" for _ in range(33)]

        # Debug:
        self.runtime = perf_counter()
        self.cameraTime = 0
        self.meshTime = 0
        self.colliderTime = 0
        self.drawTime = 0
        self.renderTime = 0

    def displayLoading(self, WindowSurface=pygame.Surface, text=''):
        """ This method displays the loading screen. """
        
        WindowSurface.blit(self.loading_background, (0, 0))
        self.text_lines.append(text)

        if len(self.text_lines) > 33: del self.text_lines[0]

        for i in range(33):
            current_text = self.text_lines[i]
            for _ in range((63 - len(current_text))):
                current_text += ' '
            display_text(WindowSurface,33-i,current_text,self.font,'c',(239,223,204),True)
        
        pygame.display.flip()

    def loadLevel(self, filename):
        """ This method loads assets from a level file. *** UNFINISHED *** """
        
        file = open(filename, "r")
        filesize = len(file)
        lineIndex = 0

        while lineIndex < filesize:
            line = file[lineIndex]
            # Strip and split line at space characters
            line = line.strip("\n") 
            line = line.split(" ")

            print(line)

    def loadFiles(self, meshes, colliders, textures, WindowSurface):
        """ This method loads meshes and textures into the game. """
        self.displayLoading(WindowSurface, "================== " + VERSION_TEXT + " ===================")
        self.displayLoading(WindowSurface, "Loading Textures:")

        # ============================= Load Textures: ============================= #

        for filename in textures:
            try:
                texture = pygame.image.load(TEXTURES + filename).convert()
                #texture = pygame.transform.flip(texture, False, True)  # I goofed up and inverted all the math, its easier to just flip the image here.
                self.displayLoading(WindowSurface,'Texture loaded, "' + filename + '".')
            except FileNotFoundError:
                texture = self.missingTexture
                self.displayLoading(WindowSurface,'Warning! Failed to load texture, "' + filename + '".')

            # Store textures with their size for quick access. Avoids getatrr() call with .getwidth() & .getheight() methods.
            texture = (texture, texture.get_width() - 1, texture.get_height() - 1)
            self.textures.append(texture)
        
        # =============================== Load Meshes: ============================= #

        self.displayLoading(WindowSurface,'Loading Meshes:')


        for i in range(len(meshes)):
            
            # Unpack mesh
            self.displayLoading(WindowSurface,'Loading Mesh, "' + meshes[i][0] + '".')
            mesh = UnpackMesh(ASSETS + meshes[i][0])

            # Read and error check texture index:
            texture_index = meshes[i][1]
            if type(texture_index) is not int:
                raise Exception('invalid texture index.')
            else:
                mesh.texIndex = meshes[i][1]            
            
            # Read and error check mesh type:
            mesh_type = str(meshes[i][4]).lower()
            if mesh_type in ["static", "dynamic"]:
                if mesh_type == "static":
                    mesh.static = True
                    # move mesh before fragmentation.
                    mesh.rotate(meshes[i][3])
                    mesh.move(meshes[i][2])
                    self.displayLoading(WindowSurface,'Fragmenting mesh...')
                    self.fragmented_meshes.extend(fragMesh(mesh, self.fragment_size))
                    self.displayLoading(WindowSurface,'Fragmenting mesh... Done!')
                else:
                    mesh.static = False
                    mesh.rotation = meshes[i][3]
                    mesh.position = meshes[i][2]
                    self.meshes.append(mesh)   
            else:
                raise Exception('invalid prop state, use "static" or "dynamic".')

        # ============================= Load colliders: ============================ #

        self.displayLoading(WindowSurface,'Loading Colliders:')

        fragmented_colliders = []
        for collider in colliders:
            filename, position, rotation, collider_type = collider[:]
            
            #Load mesh collider as Mesh class:
            self.displayLoading(WindowSurface,'Loading collider, "' + filename + '".')
            colliderMesh = UnpackMesh(COLLIDERS + filename)
            collider_type = str(collider_type).lower()
            
            if collider_type in ["static", "dynamic"]:
                if collider_type == "static":
                    colliderMesh.rotate(rotation)
                    colliderMesh.move(position)
                    self.displayLoading(WindowSurface,'Fragmenting mesh...')
                    fragmented = fragMesh(colliderMesh, self.fragment_size)
                    fragmented_colliders.extend([MeshCollider(mesh) for mesh in fragmented])
                    self.displayLoading(WindowSurface,'Fragmenting mesh... Done!')
                
                else:
                    self.colliders.append(MeshCollider(colliderMesh, position, rotation))
            
            else:
                raise Exception('invalid collider state, use "static" or "dynamic".')

        self.fragmented_colliders = fragmented_colliders

        self.displayLoading(WindowSurface,'All assets loaded, starting render...')

        # create duplicate versions of dynamic meshes and colliders. this is so that when a mesh or collider is 
        # moved, rotated, or scaled an original version can be preserved to avoid accumulating error with each transform.
        self.r_meshes = dc(self.meshes)
        self.r_colliders = dc(self.colliders)

        # Convert to tuple since these are going to be static anyway.
        self.colliders = tuple(self.colliders)
        self.meshes = tuple(self.meshes)

        self.displayLoading(WindowSurface,'Press any key to continue.')

        # After displaying loading text, wait for keyboard input.
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminate()
                elif event.type == pygame.KEYDOWN:
                    return

    def initiateBuffer(self):
        """ This function clears the depth buffer. """
        # For every horizontal line of the buffer:
        dLine = [self.camera.far for _ in range(self.camera.resolution[0])]
        
        # For every vertical line of the buffer:
        self.depth_buffer = tuple([dc(dLine) for _ in range(self.camera.resolution[1])])

    def run_logic(self, clock):
        timer = perf_counter()
        self.frameDelta = clock.tick(FPS) * 0.001 * FPS
        self.camera.update(self.frameDelta, self.r_colliders + self.fragmented_colliders)
        self.cameraTime += timer - perf_counter()

        # Check for exit condition:
        if not self.camera.active: self.terminate_game()
        
        # Update colliders meshes:
        self.update_meshes()
        self.update_colliders()

    def update_colliders(self):
        """ This method updates the position and rotation of colliders. """
        timer = perf_counter()
        
        for i in range(len(self.colliders)):
            collider, r_collider = self.colliders[i], self.r_colliders[i]
            if collider.enabled:
                # Check if rotation, position, or scale has changed since last frame:
                checks = (self.r_colliders[i].rotation != collider.rotation, self.r_colliders[i].position != collider.position)
                if True in checks:
                    self.r_colliders[i] = dc(collider)
                    r_collider = self.r_colliders[i]
                    if checks[0]:   # Collider has changed rotation:
                        r_collider.rotate(collider.rotation)    # Rotate collider.
                        r_collider.rotation = collider.rotation # Update rotation value for next frame.
                
                    else:           # Collider has changed position:
                        r_collider.move(collider.position)      # Move collider
                        r_collider.position = collider.position
        
        self.colliderTime += timer - perf_counter()
                
    def update_meshes(self):
        """ This method updates the position and rotation of meshes. """
        timer = perf_counter()
        self.culled_meshes = []
        
        #               UPDATE DYNAMIC MESHES:            #

        # First pass on meshes to quickly get rid of ones that are definitely not in view.
        checked_meshes = [i for i in range(len(self.meshes)) if meshBoundingSphereCull(self.camera ,self.meshes[i].origin, self.meshes[i].farPoint)]
        
        # Second pass on reduced list of meshes, more accurate but takes longer.
        for i in checked_meshes:
            mesh, r_mesh = self.meshes[i], self.r_meshes[i]
            #if self.meshBoundingBoxCull(dc(mesh.bounds.polygons)):
            # Check if rotation, position, or scale has changed since last frame:
            checks = (self.r_meshes[i].rotation != mesh.rotation, self.r_meshes[i].position != mesh.position)
            if True in checks:
                self.r_meshes[i] = dc(mesh)
                r_mesh = self.r_meshes[i]
                if checks[0]:   # Mesh has changed rotation
                    r_mesh.rotate(mesh.rotation)
                    r_mesh.rotateNormals(mesh.rotation)
                    r_mesh.rotation = mesh.rotation
                    mesh.bounds.update(r_mesh.vertices)
                
                else:           # Mesh has changed position
                    r_mesh.move(mesh.position)
                    r_mesh.position = mesh.position
                
            self.culled_meshes.append(r_mesh)

        #               UPDATE STATIC MESHES:             #
        # I used list comprehensions here because it's faster to use the.extend method over appending one at a time.
        # I expect there to be far more fragmented meshes than dynamic ones so efficiency is key here.
        
        # Done in two passes just like for the dynamic meshes.
        checked_frag_meshes = [mesh for mesh in self.fragmented_meshes if meshBoundingSphereCull(self.camera, mesh.origin, mesh.farPoint)]
        self.culled_meshes.extend([mesh for mesh in checked_frag_meshes if meshBoundingBoxCull(self.camera, dc(mesh.bounds.polygons))])
        
        self.meshTime += timer - perf_counter()


    def mesh_occlusion_cull(self, mesh):
        """ This method checks if the area a mesh is to be rendered too can be seen by the camera. """
        

    def fast_camera_projection(self, mesh):
        """ This method transforms the vertices of a mesh from world space to screen space. """
        hs, vs, hc, vc = self.camera.h_scale_const, self.camera.v_scale_const, self.camera.h_center, self.camera.v_center
        mesh.vertices = [v.update(((v.x * inv_z) * hs) + hc, ((v.y * inv_z) * vs) + vc, inv_z) for v, inv_z in zip(mesh.vertices, [-1 / (v.z + 0.001) for v in mesh.vertices])]
    
    def render_skybox(self):
        """ This method handles drawing the sky box on top of the rendered screen. """
        # Create a copy to preserve the original mesh.
        skybox = dc(self.skybox)                

        # Rotate sky box such that it appears fixed in rotation:
        skybox.rotate_y(self.camera.y_rotation)
        skybox.rotate_x(self.camera.x_rotation)

        # Clip skybox to view:
        clipMesh(self.camera.clip,skybox)
        self.fast_camera_projection(skybox)
        RasterizeSkybox(self.pixelArray, self.screen, skybox, self.skytex)

    def render(self, Display):
        """ This function renders the scene to a surface. """
        timer = perf_counter()
        
        # Clear screen & depth buffer, copy over culled meshes:
        if RENDER_SKY: self.pixelArray[:] = (255, 0, 255)
        depth_buffer = dc(self.depth_buffer)
        meshes = dc(self.culled_meshes)

        # Initialize texture data:
        texIndex = 0                 # Default to texture No.0
        t = self.textures[texIndex]  # Get texture data for current image.

        # Sort meshes by distance to camera, draw the nearest ones first to avoid overdrawing.
        distances = QuickSort([(m.origin - self.camera.position).magnitude_squared() for m in meshes], [i for i in range(len(meshes))])[1]

        self.renderTime += timer - perf_counter()
        timer = perf_counter()
        
        for i in distances:
            mesh = meshes[i]
            
            # Remove back-facing Faces:
            index = 0
            while index < len(mesh):
                if mesh[index][3].dot((mesh[index][0][0] - self.camera.position)) > 0: del mesh[index]
                else: index += 1
            clipMesh(self.camera.r_clip, mesh)
            if mesh.polygons != []:  # this is effectively the same as if len(mesh) != 0.
                mesh.move(-self.camera.position)
                mesh.rotate_y(self.camera.y_rotation)
                mesh.rotate_x(self.camera.x_rotation)
                self.fast_camera_projection(mesh)
                mesh.getDistance()
                # Check for change in texture index, sort polygons then render:
                if mesh.texIndex != texIndex: texIndex, t = mesh.texIndex, self.textures[mesh.texIndex]
                indecies = QuickSort(mesh.depth, [i for i in range(len(mesh))])[1]
                RasterizePolygon(self.pixelArray, depth_buffer, mesh.polygons, t, indecies)

        self.drawTime += timer - perf_counter()
        timer = perf_counter()
        
        # Draw sky box:
        if RENDER_SKY and TRANSPARENCY in self.pixelArray: 
            self.render_skybox()

        # Scale to screen size:
        scaled = pygame.transform.scale(self.screen, (WIDTH, HEIGHT))
        Display.blit(scaled,(0,0))
        
        # Draw debug text:
        display_text(Display, 0, "FPS  | " + str(round((FPS / self.frameDelta))), self.font,'l',(128,128,128),True)
        display_text(Display, 1, "MDL# | " + str(len(meshes)), self.font,'l',(128,128,128),True)
        display_text(Display, 2, "POS  | " + str(round(self.camera.position[0], 4)) + " / " + str(round(self.camera.position[1], 4)) + " / " + str(round(self.camera.position[2], 4)), self.font,'l',(128,128,128),True)
        
        # Update display:
        pygame.display.flip()
        self.renderTime += timer - perf_counter()

    def terminate_game(self):
        runtime = self.runtime - perf_counter()

        times =     [int((round(self.meshTime / runtime, 2)) * 100),
                     int((round(self.colliderTime / runtime, 2)) * 100),
                     int((round(self.cameraTime / runtime, 2)) * 100),
                     int((round(self.drawTime / runtime, 2)) * 100),
                     int((round(self.renderTime / runtime, 2)) * 100),]
        
        total = sum(times)
        lost_time = 100 - total
        
        print("#### Render Stopped! ####\n")
        print("Mesh update time:\t",    times[0],end="%\n")
        print("Collider update time:\t",times[1],end="%\n")
        print("Camera time:\t\t",       times[2],end="%\n")
        print("Draw time:\t\t",         times[3],end="%\n")
        print("Render time:\t\t",       times[4],end="%\n")
        print("Misc losses:\t\t",       lost_time,end="%\n")
        print("\n\n#### Render Stopped! ####")
        
        terminate()


def main():
    pygame.init()
    pygame.mouse.set_visible(False)
    
    clock = pygame.time.Clock()
    Display = pygame.display.set_mode((WIDTH, HEIGHT),vsync=0)
    icon = pygame.image.load("dependencies/textures/icon.png").convert()

    pygame.display.set_caption(VERSION_TEXT)
    pygame.display.set_icon(icon)

    instance = Game()
    
    #NOTE:
    # the forward axis is -x

    textures = (
        'ship.png',
        #'Axis.png',
    )
    
    props = (
        ('ship.obj',    0,  Vect3(0,0,0), (0,0,0), 'dynamic'), 
        #('Axis.obj',    1,  Vect3(0,0,0), (0,0,0), 'dynamic'),
    )
 
    colliders = (
        ('ship.obj',    Vect3(0,0,0),(0,0,0),'static'),
    )
                 
    instance.loadFiles(props, colliders, textures, Display)
    i = 0

    while 1:
        
        instance.run_logic(clock)
        instance.render(Display)


main()
