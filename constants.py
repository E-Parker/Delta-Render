# This program was writen by Ethan Parker.

import sys
import pygame


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# DEFINING CONSTANTS AND SYSTEM FUNCTIONS

# Display:
WIDTH, HEIGHT = 640, 480    # Screen Size.
FOV = 120                   # Camera Felid of view, breaks after 170 degrees.
FPS = 60                    # Refresh rate. (max of 60 due to Pygame's render pipeline)
RENDER_SCALE = True         # Bool for rendering sky box
VERSION_TEXT =  "DELTA-RENDER v7 patch 0.2.0"

R_WIDTH = 128
R_HEIGHT = 96

# Camera:
RENDER_SKY = True 
NEAR_CLIP = 0.001       # Near clipping distance.
FAR_CLIP = 64           # Far clipping distance.
M_SPEED = 0.1           # Default movement speed.
AIR_DECAY = 0.98        # Velocity decay rate while airborne.
GROUND_DECAY = 0.70     # Velocity decay rate while on the ground.

SENSITIVITY = 25
R_SENSITIVITY = SENSITIVITY * 0.001

# Collision:
COLLIDER_SIZE = 1		                # Default size for the collider.
MESH_COLLIDER_GAP = COLLIDER_SIZE * 0.5 # Gap between points for edge interpolation. One half seems stable enough.
FRAGMENT_SIZE = 16

# Colours:
LIGHT_BIAS = 0.5
DARK = pygame.Color(5, 10, 25)
LIGHT = pygame.Color(255, 253, 248)
TRANSPARENCY = pygame.Color(255, 0, 255)

# Path:
DEP_TEXTURES =  	"dependencies/textures/"
DEP_ASSETS =    	"dependencies/meshes/"
TEXTURES =      	"assets/textures/"
ASSETS =        	"assets/meshes/"
SCRIPT =        	"assets/scripts/"
COLLIDERS =     	"assets/colliders/"
SKYBOX =        	"skybox.obj"
MISSING_OBJECT =	"missing_object.obj"
SKYTEX =        	"skybox_default.png"

# Math:
DEG_TO_FLOAT = 1/360            # Convert degrees to a [0-1] float.
ONE_THIRD = 1/3                 # divide by 3 constant
LIGHTING = (0.20, -0.65, 0.10)  # Vector for angle of lighting. MUST BE A UNIT VECTOR.
SORT_LIMIT = 1000


def terminate():
    pygame.quit()
    sys.exit()
