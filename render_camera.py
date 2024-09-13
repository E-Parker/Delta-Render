# This program was writen by Ethan Parker
# This program handles all functions relating to the Camera class.

from math import *
from collision import *
from constants import *
from pygame import Vector3 as Vect3
from pygame import Vector2 as Vect2
from copy import deepcopy as dc

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

        # Camera Clipping:
        self.near = near                # used to define near clip plane.
        self.far = far                  # used to define far clip plane.
        self.maxDist = far * far        # check if an object is within view, no point in clipping something you can't see!
        self.clip = self.genClip()      # Generate planes from peramiters set above.

        # Boolens & States:
        self.active = True  # Bool for game pause / exit condition
        self.move = [False, False, False, False, False, False]  # bools for if camera is moving and in which direction.
        
        # Rotation:
        self.rotation_max = 80      # Maximum vertical rotation in degrees 
        self.x_rotation = 0
        self.y_rotation = 0
        self.z_rotation = 0

        # Movement:
        self.gravity = Gravity()
        self.mSpeed = moveSpeed                 # movement speed multiplier
        self.maxSpeed = self.mSpeed * 2         # Max movement speed
        self.maxAirSpeed = self.maxSpeed * 2    # Max movement speed in air
        self.aceeleration = 0.40                # Amount of control over change in velocity
        self.airAceeleration = 0.10             # Amount of control while airborne
        self.radius = coliderSize               # colider radius
        self.height = Vect3(0,0.3,0)            # Height offset
        self.airborne = True                    # Bool for if camera is in the air

        # Vectors:
        self.d_rotation =   Vect3(0, 0, 1)      # Vector for checking if a face is on screen, (rotation vector scaled by aspect ratio)
        self.position =     Vect3(0, 0, 0)      # Position of the camera.
        self.rotation =     Vect3(0, 0, 1)      # Unit vector in the direction the camera is facing.
        self.velocity =     Vect3(0, 0, 0)      # Velocity of the camera.

        # Debug:
        self.noclip = True
        self.allow_movement = True

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
        rgt = Plane(c5, c6, c8)  # Right 

        # planes stored in this order so the game has the least amount of work to do per clip operation.
        planes = (ner, lft, rgt, top, btm, far)

        return planes

    def update(self, frameDelta, colliders):
        """ This function handles updating the camera and it's properties. """

        # offset position by height:
        self.position -= self.height

        # calculate friction depending on if the camera is in the air or on the ground
        if self.airborne and not self.noclip:
            self.velocity = self.velocity * AIR_DECAY
        else:
            self.velocity = self.velocity * GROUND_DECAY
        
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
            vel = (self.velocity + Vect3(0, 0.1 * self.gravity.current_value, 0)) * frameDelta
            
            # Check for Collisions:
            walls = 0
            floors = 0
            pos = self.position + vel
            
            for collider in colliders:
                if collider.enabled:
                    collision = collider.CollideMesh(self.position, pos, self.radius)
                    pos = collision[0]
                    walls += collision[1]
                    floors += collision[2]

            self.position = pos
 
            if floors == 0:
                self.gravity.set_falling()  # if not colliding with the floor, character is falling.
                self.airborne = True
            else:
                self.gravity.current_time = 7
                self.airborne = False

        # Get mouse position
        mouse_position = pygame.mouse.get_pos()
        pygame.mouse.set_pos((self.h_screen_center, self.v_screen_center))

        # Update Vectors:
        y_rotation = degrees((mouse_position[0] - self.h_screen_center) * R_SENSITIVITY)
        x_rotation = degrees((mouse_position[1] - self.v_screen_center) * R_SENSITIVITY)
        
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

