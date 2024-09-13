# This program was written by Ethan Parker
# This program consolidates all the functions related to rendering to the screen.

from constants import *
from render_math import *

# The interpLine functions are basically unreadable. It sucks but I have to squeeze every last ounce of performance out of this
# if it's going to run in real time. It sucks but this is how it is.

def interpLine(a, b):
    """ This function is intended for rasterization and handles generating a line segment from x1,y1,z1 to x2,y2,z2. """
    a, u = a[:2]
    b, v = b[:2]
    start, end = round(a.y), round(b.y)
    if start == end: 
        return []
    ui, vi, inv = u * a.z, v * b.z, 1 / (b.y - a.y)
    ax1, az1, uix, uiy = a.x, a.z, ui.x, ui.y
    ax, az, ux, uy = (b.x - a.x) * inv, (b.z - a.z) * inv, (vi.x - ui.x) * inv, (vi.y - ui.y) * inv
    line = [(ax1, az1, uix, uiy,)]
    line.extend([((ax1 + ax * y), az1 + az * y, uix + (ux * y), uiy + (uy * y),) for y in range(1, (end - start), 1)])
    return line


def interpLineBrightness(a, b):
    """ This function is intended for rasterization and handles generating a line segment from x1,y1,z1,b1 to x2,y2,z2,b2. """
    a, u, ab = a[:3]
    b, v, bb = b[:3]
    start, end = round(a.y), round(b.y)
    l = end - start
    if start == end: 
        return []
    ab, bb = ab * a.z, bb * b.z
    ui, vi, inv = u * a.z, v * b.z, 1 / (b.y - a.y)
    ax1, az1, uix, uiy = a.x, a.z, ui.x, ui.y
    ax, az, ux, uy, br = (b.x - a.x) * inv, (b.z - a.z) * inv, (vi.x - ui.x) * inv, (vi.y - ui.y) * inv, (bb - ab) * inv
    line = [(ax1, az1, uix, uiy, ab)]
    line.extend([((ax1 + ax * y), az1 + az * y, uix + (ux * y), uiy + (uy * y), ab + (br * y)) for y in range(1, l, 1)])
    return line


def RasterizePolygon(pixel_array, depth, polygons, texture, indecies):
    texture, txW, txH = texture
    txW_max, txH_max = txW - 1, txH - 1

    # Define lambda functions for sampling texture coordinates.
    distance, z = 0, 0
    getPixelWrap = lambda x, xl, y, yl:  ((int((x + xl * distance) * z * txW) % txW),(int((y + yl * distance) * z * txH) % txH))
    getPixelClamp = lambda x, xl, y, yl: (clamp(int((x + xl * distance) * z * txW),1,txW_max),clamp(int((y + yl * distance) * z * txH),1,txH_max))

    try:
        for index in indecies:
            a, b, c= polygons[index][:3]

            # Sort points by top to bottom:
            if a[0].y > b[0].y: a, b = b, a
            if b[0].y > c[0].y: b, c = c, b
            if a[0].y > b[0].y: a, b = b, a

            # Generate left and right sides of the triangle:
            r_side = interpLineBrightness(a, c)
            height = len(r_side)
            
            if height == 0:
                continue
            l_side = interpLineBrightness(a, b) + interpLineBrightness(b, c)
            y_offset = a[0].y
            
            # Sort left and right:
            middle = height // 2
            if l_side[middle][0] < r_side[middle][0]:
                l_side, r_side = r_side, l_side
        
            for lclY in range(height):
                # Assign variables:
                rsx, rsz, trsx, trsy, rsb = r_side[lclY]
                lsx, lsz, tlsx, tlsy, lsb = l_side[lclY]

                # Check that scanline has length: 
                if rsx == lsx:
                    continue

                # Lots of these are defined here because its faster to look it up in memory than to recalculate it.
                r_side_offset, l_side_offset, y = round(rsx), round(lsx), round(lclY + y_offset)
                xlerp, ylerp, zlerp, blerp = trsx - tlsx, trsy - tlsy, rsz - lsz, rsb - lsb
                inv_distance, dLine, line, line_len, start = 1 / (rsx - lsx), depth[y], [], 0, r_side_offset
                
                # Build scan line:
                for x in range(r_side_offset, l_side_offset):
                    distance = (x - l_side_offset) * inv_distance    # Precalculate part of the interpolation formula.
                    inv_z = (lsz + zlerp * distance)  # Find the current z value at this pixel.
                    
                    if dLine[x] > inv_z:  # Check depth buffer against the current z value.
                        line_len += 1
                        z = 1 / inv_z                 # solve for actual z (since 1/1/z = z).
                        albido = texture.get_at(getPixelClamp(tlsx,xlerp,tlsy,ylerp))
                        if albido == TRANSPARENCY:
                            colour = pixel_array[x, y]
                        else:
                            # write z value to the depth buffer
                            dLine[x] = inv_z   
                            # interpolate the lighting for this pixel.
                            brightness = (lsb + (blerp * distance)) * z
                            if brightness > 0:  lighting = LIGHT
                            else:               lighting = DARK
                            colour = albido.lerp(lighting, cos((1 - abs(brightness) )* pi) * 0.25 + 0.25)        
                        line.append(colour)
                    else:  # the line has been interupted, write data and start over.
                        if start != x: 
                            pixel_array[start:x, y] = line
                            line, line_len = [], 0  
                        start = x + 1
                if start != l_side_offset and line_len == l_side_offset - start:
                    pixel_array[start:l_side_offset, y] = line
    except:
        pass
    

def RasterizeSkybox(pixel_array, surface, mesh, texture):
    try:
        maped = surface.map_rgb(TRANSPARENCY)
        texture, txW, txH = texture
        for a, b, c, n in mesh:

            # Sort points by top to bottom:
            if a[0].y > b[0].y: a, b = b, a
            if b[0].y > c[0].y: b, c = c, b
            if a[0].y > b[0].y: a, b = b, a
            
            # Generate left and right sides of the triangle:
            right_edge = interpLine(a, c)
            
            height = len(right_edge)
            if height == 0: 
                continue      
            
            left_edge = interpLine(a, b)
            left_edge.extend(interpLine(b, c))
            y_offset = a[0].y

            # Sort left from right:
            middle = height // 2
            if left_edge[middle][0] < right_edge[middle][0]: 
                left_edge, right_edge = right_edge, left_edge

            for relative_y in range(height):
                # Assign variables:
                rsx, rsz, trsx, trsy = right_edge[relative_y]
                lsx, lsz, tlsx, tlsy = left_edge[relative_y]
                
                # Check for valid scan line:
                if rsx == lsx:
                    continue

                # Lots of these are defined here because its faster to look it up in memory.
                offset_left, offset_right, y = round(rsx), round(lsx), clamp(round(relative_y + y_offset), 0, R_HEIGHT - 1)
                xlerp, ylerp, zlerp = trsx - tlsx, trsy - tlsy, rsz - lsz
                i_distance, line, line_len, start = 1 / (rsx - lsx), [], 0, offset_left
                
                # Build scanline:
                for x in range(max((offset_left,0)), min((offset_right,R_WIDTH))):
                    if pixel_array[x][y] == maped:
                        line_len += 1
                        step = (x - offset_right) * i_distance
                        z = 1 / (lsz + zlerp * step)
                        colour = texture.get_at((((int((tlsx + xlerp * step) * z * txW)) % txW),
                                                    ((int((tlsy + ylerp * step) * z * txH)) % txH)))
                        line.append(colour)
                    else:  # the line has been interrupted, write data and start over from new location.
                        if start != x: pixel_array[start:x, y], line, line_len = line, [], 0
                        start = x + 1
                if start != offset_right and line_len == offset_right - start:
                    pixel_array[start:offset_right, y] = line
    except:
        pass
