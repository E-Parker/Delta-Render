# This program was writen by Ethan Parker.

import sys
import pygame


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# DEFINING CONSTANTS AND SYSTEM FUNCTIONS

# Display:
WIDTH, HEIGHT = 640, 480   # Screen Size.
RENDERSCALE = 0.2          # Determins the render resolution.
FOV = 90                   # Camera Feild of view, breaks after 170 degrees.
FPS = 60                   # Refresh rate. (max of 60 due to Pygame's render pipeline)
RENDERSKY = True           # Bool for rendering sky box

RWIDTH = int(WIDTH * RENDERSCALE)
RHEIGHT = int(HEIGHT * RENDERSCALE)

# Camera:
NEARCLIP = 0.01		    # Near clipping distance.
FARCLIP = 512           # Far clipping distance.
MSPEED = 0.04           # Default movement speed.
COLIDER_SIZE = 0.6		# Default size for the colider.

# Colours:
DARK = pygame.Color(0, 10, 40)
LIGHT = pygame.Color(255, 240, 230)
MAXLIGHTING = 0.15
LIGHTINGBIAS = 0.3
TRANSPARENCY = pygame.Color(255, 0, 255)

# Path:
DD_TEXTR =  "dependencies/textures/"
DD_ASSET =  "dependencies/meshes/"
TEXTR =     "assets/textures/"
ASSET =     "assets/meshes/"
CLLDR =     "assets/colliders/"
SKYBOX =    "skybox.obj"
SKYTEX =    "skybox_default.png"


# Math:
DEGTOFLOAT = 1/360              # Convert degrees to a [0-1] float.
ONETHIRD = 1/3                  # divide by 3 constant
LIGHTING = (0.15, -0.65, 0.15)  # Vector for angle of lighting. MUST BE A UNIT VECTOR.
SORTLIMIT = 500


def terminate():
    pygame.quit()
    sys.exit()
