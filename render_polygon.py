# This program was writen by Ethan Parker
# This program consolidates all the functions related to rendering to the screen.

from constants import *
from render_math import *


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
    """ This function is intended for rasterization and handles generating a line segment from x1,y1,z1 to x2,y2,z2. """
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


def RasterizeMesh(pixel_array, depth, mesh, texture, indecies):
    texture, txW, txH = texture
    for index in indecies:
        a, b, c, n = mesh[index]

        # Sort points by top to bottom:
        if a[0].y > b[0].y: a, b = b, a
        if b[0].y > c[0].y: b, c = c, b
        if a[0].y > b[0].y: a, b = b, a

        # Generate left and right sides of the triangle:
        r_side = interpLineBrightness(a, c)
        height = len(r_side)
        
        if height == 0:
            continue
        l_side = interpLineBrightness(a, b)
        l_side.extend(interpLineBrightness(b, c))
        y_offset = round(a[0].y)
        
        # Sort left and right:
        middle = height // 2
        if l_side[middle][0] < r_side[middle][0]:
            l_side, r_side = r_side, l_side
        try:
            for lclY in range(height):
                # Assign variables:
                rsx, rsz, trsx, trsy, rsb = r_side[lclY]
                lsx, lsz, tlsx, tlsy, lsb = l_side[lclY]

                # Check that scanline has length: 
                if rsx == lsx:
                    continue

                # Lots of these are defined here because its faster to look it up in memory than to recalculate it.
                r_side_offset, l_side_offset, y = round(rsx), round(lsx), lclY + y_offset
                xlerp, ylerp, zlerp, blerp = trsx - tlsx, trsy - tlsy, rsz - lsz, rsb - lsb
                inv_distance, dLine, line, start = 1 / (rsx - lsx), depth[y], [], r_side_offset
                
                # Build scan line:
                for x in range(r_side_offset, l_side_offset):
                    lclX = x - l_side_offset
                    distance = lclX * inv_distance    # Precalculate part of the interpolation formula.
                    inv_z = (lsz + zlerp * distance)  # Find the current z value at this pixel.
                    if dLine[x] > inv_z:
                        z = 1 / inv_z                 # solve for actual z (since 1/1/z = z).
                        colour = texture.get_at(((txW - int((tlsx + xlerp * distance) * z * txW) % txW),
                                                 (txH - int((tlsy + ylerp * distance) * z * txH) % txW)))

                        if colour != TRANSPARENCY: 
                            dLine[x] = inv_z  # write z value to the depth buffer
                            light = (lsb + (blerp * distance)) * z  # interpolate the lighting for this pixel.
                            if light < 0:                           # positive lighting means pixel is in shadow.
                                l = max((min((0 - light, 0.25)),0))
                                line.append(colour.lerp(LIGHT,l))
                            else:
                                l = max((min((light, 0.9)),0))
                                line.append(colour.lerp(DARK,l))
                        else:
                            line.append(pixel_array[x, y])  
                    else:  # the line has been interupted, write data and start over after starting
                        if start != x: 
                            pixel_array[start:x, y] = line
                            line = []   
                        start = x + 1
                if start != l_side_offset and len(line) == l_side_offset - start:
                    pixel_array[start:l_side_offset, y] = line
        except:
            pass


def RasterizeSkybox(pixel_array, surface, mesh, texture, trancparency):
    maped = surface.map_rgb(trancparency)
    texture, txW, txH = texture
    for a, b, c, n in mesh:

        # Sort points by top to bottom:
        if a[0].y > b[0].y: a, b = b, a
        if b[0].y > c[0].y: b, c = c, b
        if a[0].y > b[0].y: a, b = b, a
        
        # Generate left and right sides of the triangle:
        r_side = interpLine(a, c)
        
        height = len(r_side)
        if height == 0: 
            continue      
        
        l_side = interpLine(a, b)
        l_side.extend(interpLine(b, c))
        y_offset = round(a[0].y)

        # Sort left from right:
        middle = height // 2
        if l_side[middle][0] < r_side[middle][0]: 
            l_side, r_side = r_side, l_side
        
        try:
            for lclY in range(height):
                # Assign variables:
                rsx, rsz, trsx, trsy = r_side[lclY]
                lsx, lsz, tlsx, tlsy = l_side[lclY]
                
                # Check for valid scan line:
                if rsx == lsx:
                    continue

                # Lots of these are defined here because its faster to look it up in memory.
                r_side_offset, l_side_offset, y = round(rsx), round(lsx), lclY + y_offset
                xlerp, ylerp, zlerp = trsx - tlsx, trsy - tlsy, rsz - lsz
                inv_distance, line, start = 1 / (rsx - lsx), [], r_side_offset
                
                # Build scanline:
                for x in range(r_side_offset, l_side_offset):
                    if pixel_array[x][y] == maped:
                        distance = (x - l_side_offset) * inv_distance
                        z = 1 / (lsz + zlerp * distance)
                        colour = texture.get_at(((txW - round((tlsx + xlerp * distance) * z * txW) % txW),
                                                 (txH - round((tlsy + ylerp * distance) * z * txH) % txH)))
                        line.append(colour)
                    else:  # the line has been interrupted, write data and start over from new location.
                        if start != x: 
                            pixel_array[start:x, y] = line
                            line = []
                        start = x + 1
                if start != l_side_offset and len(line) == l_side_offset - start:
                    pixel_array[start:l_side_offset, y] = line
        except:
            pass

