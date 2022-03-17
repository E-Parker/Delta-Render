# This program handles rasterization and drawing of polygons for version 2.8 of ΔRender
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

DITHER = (((0 / 16) - 0.5,(8 / 16) - 0.5,(2 / 16) - 0.5,(10 / 16) - 0.5),
		((12 / 16) - 0.5,(4 / 16) - 0.5,(14 / 16) - 0.5,(6 / 16) - 0.5),
		((3 / 16) - 0.5,(11 / 16) - 0.5,(1 / 16) - 0.5,(9 / 16) - 0.5),
		((15 / 16) - 0.5,(7 / 16) - 0.5,(13 / 16) - 0.5,(5 / 16) - 0.5))


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
	# This is done so that when we interpolate between z values, they'll be in screen-space (1 / z).
	ui, vi = u * a.z, v * b.z
	line = ((a, ui),)

	# interpolate points from start to end:
	for y in range(start + 1, end, 1):
		# generate new position
		point = Vect3(a.x + ((b.x - a.x) * (y - a.y)) * inv_dist, y, a.z + ((b.z - a.z) * (y - a.y)) * inv_dist)
		tx_point = Vect2(ui.x + ((vi.x - ui.x) * (y - a.y)) * inv_dist, ui.y + ((vi.y - ui.y) * (y - a.y)) * inv_dist)
		line += ((point, tx_point,),)
	return line


def render(mesh, screen, depth, textures, ti, filtering):
	""" This function handles rendering a mesh to the screen. """
	if len(mesh) == 0: return

	indices = list(range(len(mesh)))
	indices = func.QuickSort(mesh.depth, indices)[1]

	if filtering: dither = DITHER
	else: dither = ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0),)

	current_ti = ti[0]
	texture = textures[current_ti]
	txH, txW = texture.get_height() - 1, texture.get_width() - 1

	for index in indices:

		if current_ti != ti[index]:
			current_ti = ti[index]
			texture = textures[current_ti]
			txH, txW = texture.get_height() - 1, texture.get_width() - 1

		face = mesh[index]
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
			ls, rs= left[y][0], right[y][0]
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
							z = 1 / inv_z  # Convert z back to world space, solving for the z at this pixel. ( 1 / (1 / z) = z)
							# Get current texture coordinate:
							d = dither[point[0] % 4][point[1] % 4]
							u = (func.interp(left[y][1].x, ls.x, right[y][1].x, rs.x, step) * z) % 1
							v = (func.interp(left[y][1].y, ls.x, right[y][1].y, rs.x, step) * z) % 1
							uv = (int(u * txW + d), int(v * txH + d))
							colour = texture.get_at(uv)

							if colour != TRANSPARENCY:
								screen[point[0]][point[1]] = colour
								depth[point[1]][point[0]] = inv_z
					except IndexError:
						pass


def renderSkybox(mesh, screen, pixel_array, texture):
	""" This function handles rendering a mesh to the screen. """

	maped = screen.map_rgb(TRANSPARENCY)
	txH, txW = texture.get_height() - 1, texture.get_width() - 1

	for index in range(len(mesh)):
		face = mesh[index]
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
		for local_y in range(height):
			# Render with dithering & depth buffer
			ls, rs = left[local_y][0], right[local_y][0]
			offset = round(rs.x)
			dist = (rs.x - ls.x)
			if dist != 0:
				inv_dist = 1 / dist
				for local_x in range(round(ls.x) - offset):
					x, y = local_x + offset, local_y + top
					try:
						if pixel_array[x][y] == maped:
							dither = DITHER[y % 4][x % 4]
							z = 1 / quickInterp(ls.z, ls.x, rs.z, inv_dist, x)
							u = (func.interp(left[local_y][1].x, ls.x, right[local_y][1].x, rs.x, x) * z) % 1
							v = (func.interp(left[local_y][1].y, ls.x, right[local_y][1].y, rs.x, x) * z) % 1
							uv = (int(u * txW + dither), int(v * txH + dither))
							colour = texture.get_at(uv)
							pixel_array[x][y] = colour
					except IndexError:
						pass
