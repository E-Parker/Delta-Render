# This program handles rasterization and drawing of polygons for version 2.7 of ΔRender
# This program was writen by Ethan Parker on 03/08/22
import concurrent.futures
from ast import Index
import func
import math
import pygame
import threading
from pygame.math import Vector3 as Vect3
from pygame.math import Vector2 as Vect2

TRANSPARENCY = pygame.Color(255, 0, 255)

DITHER = ((0 / 16, 8 / 16, 2 / 16, 10 / 16),
		  (12 / 16, 4 / 16, 14 / 16, 6 / 16),
		  (3 / 16, 11 / 16, 1 / 16, 9 / 16),
		  (15 / 16, 7 / 16, 13 / 16, 5 / 16),)


def quickInterp(x1, y1, x2, inv_dist, y):
	""" This function interpolates between two points at a given y."""
	try:
		result = x1 + ((x2 - x1) * (y - y1)) * inv_dist
		return result
	except ZeroDivisionError:
		return x1


def interp_line(a, b):
	""" This function is intended for rasterization and handles generating a line segment from x1,y1,z1 to x2,y2,z2.
	Y values will be treated as an int."""
	# Check that interpolation is even possible:
	a, b, u, v = a[0], b[0], a[1], b[1]
	start, end = round(a.y), round(b.y)
	if start - end == 0:
		return ()

	# Get Inverse of distance between a and b:
	inv_dist = 1 / (b.y - a.y)

	# Multiply texture coordinate by inverted Z.
	# This is done so that correct interpolation can be done in screen-space.
	ui, vi = u * a.z, v * b.z
	line = ((a, ui),)

	# interpolate points from start to end:
	for y in range(start + 1, end, 1):
		# generate new position
		point = Vect3(a.x + ((b.x - a.x) * (y - a.y)) * inv_dist, y, a.z + ((b.z - a.z) * (y - a.y)) * inv_dist)
		tx_point = Vect2(ui.x + ((vi.x - ui.x) * (y - a.y)) * inv_dist, ui.y + ((vi.y - ui.y) * (y - a.y)) * inv_dist)
		line += ((point, tx_point,),)
	return line


def render(mesh, screen, depth, textures, textures_index, filtering):
	""" This function handles rendering a mesh to the screen. """

	indices = list(range(len(mesh)))
	indices = func.QuickSort(mesh.depth, indices)[1]

	for index in indices:
		face, texture = mesh[index], textures[textures_index[index]]
		txH, txW = texture.get_height() - 1, texture.get_width() - 1
		a, b, c = (face.a, face.u), (face.b, face.v), (face.c, face.w)

		# Sort by top to bottom:
		if a[0].y > b[0].y: a, b = b, a
		if b[0].y > c[0].y: b, c = c, b
		if a[0].y > b[0].y: a, b = b, a

		# Generate left and right sides of the polygon:
		left = interp_line(a, b) + interp_line(b, c)
		right = interp_line(a, c)

		height = len(right)
		top = round(a[0].y)

		# Check for zero area:
		if height == 0:
			continue
		middle = height // 2
		if left[middle][0].x < right[middle][0].x:
			left, right = right, left

		# Rasterize Polygon
		for y in range(height):
			# Render with dithering & depth buffer
			ls, rs, tls, trs = left[y][0], right[y][0], left[y][1], right[y][1]
			offset = round(rs.x)
			dist = (rs.x - ls.x)

			if dist != 0:
				inv_dist = 1 / dist
				for x in range(round(ls.x) - offset):
					step = x + offset
					point = (step, y + top)
					inv_z = quickInterp(ls.z, ls.x, rs.z, inv_dist, step)
					try:
						if depth[point[1]][point[0]] > inv_z:
							# Get current texture coordinate:
							z = 1 / inv_z
							if filtering: dither = DITHER[point[1] % 4][point[0] % 4]
							else: dither = 0
							u = (func.interp(tls.x, ls.x, trs.x, rs.x, step) * z) % 1
							v = (func.interp(tls.y, ls.x, trs.y, rs.x, step) * z) % 1
							uv = (int(u * txW + dither), int(v * txH + dither))
							colour = texture.get_at(uv)
							# Write to screen:
							if colour != TRANSPARENCY:
								screen[point[0]][point[1]] = colour
								depth[point[1]][point[0]] = inv_z
					except:
						pass


def renderSkybox(mesh, screen, pixel_array, texture):
	""" This function handles rendering a mesh to the screen. """
	txH, txW = texture.get_height() - 1, texture.get_width() - 1
	for i in range(len(mesh)):
		face = mesh[i]
		a, b, c, u, v, w = face.a.p, face.b.p, face.c.p, face.a.t, face.b.t, face.c.t

		# Sort by top to bottom:
		if a.y > b.y: a, b, u, v, = b, a, v, u
		if b.y > c.y: b, c, v, w, = c, b, w, v
		if a.y > b.y: a, b, u, v, = b, a, v, u

		left = interp_line(a, b, u, v) + interp_line(b, c, v, w)
		right = interp_line(a, c, u, w)
		height = len(right)
		top = round(a.y)

		# Check for zero area:
		if height == 0:
			continue
		middle = height // 2
		if left[middle][0].x < right[middle][0].x:
			left, right = right, left

		# Rasterize Polygon
		for local_y in range(height):
			# Render with dithering & depth buffer
			ls, rs, tls, trs = left[local_y][0], right[local_y][0], left[local_y][1], right[local_y][1]
			rs_offset = round(rs.x)
			ls_offset = round(ls.x)
			dist = (rs.x - ls.x)

			if dist != 0:
				inv_dist = 1 / dist

				for local_x in range(ls_offset - rs_offset):
					x = local_x + rs_offset
					y = local_y + top
					try:
						if pixel_array[x][y] == screen.map_rgb(TRANSPARENCY):
							z = 1 / quickInterp(ls.z, ls.x, rs.z, inv_dist, x)
							dither = DITHER[y % 4][x % 4]
							u = (func.interp(tls.x, ls.x, trs.x, rs.x, x) * z) % 1
							v = (func.interp(tls.y, ls.x, trs.y, rs.x, x) * z) % 1
							uv = (int(u * txW + dither), int(v * txH + dither))
							colour = texture.get_at(uv)
							pixel_array[x][y] = colour
					except IndexError:
						pass
