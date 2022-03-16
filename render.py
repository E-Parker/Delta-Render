# This program handles rasterization and drawing of polygons
# This program was writen by Ethan Parker on 03/08/22

from ast import Index
import func
import math
import pygame
from pygame.math import Vector3 as Vect3
from pygame.math import Vector2 as Vect2

DITHER = ((0 / 16, 8 / 16, 2 / 16, 10 / 16),
		  (12 / 16, 4 / 16, 14 / 16, 6 / 16),
		  (3 / 16, 11 / 16, 1 / 16, 9 / 16),
		  (15 / 16, 7 / 16, 13 / 16, 5 / 16),)


def getUV(tls, trs, ls, rs, z, step):
	u = (func.interp(tls.x, ls.x, trs.x, rs.x, step) * z) % 1
	v = (func.interp(tls.y, ls.x, trs.y, rs.x, step) * z	) % 1
	return Vect2(u, v)


def quickInterp(x1, y1, x2, inv_dist, y):
	""" This function interpolates between two points at a given y."""
	try:
		result = x1 + ((x2 - x1) * (y - y1)) * inv_dist
		return result
	except ZeroDivisionError:
		return x1


def interp_line(a, b, u, v):
	""" This function is intended for rasterization and handles generating a line segment from x1,y1,z1 to x2,y2,z2.
	Y values will be treated as an int."""
	# Check that interpolation is even possible:
	dist = (b.y - a.y)
	if dist == 0: return ()

	inv_dist = 1 / dist

	# Multiply texture coordinate by inverted Z
	ui, vi = u * a.z, v * b.z
	line = ((a, ui),)

	# interpolate points from start to end:
	for y in range(round(a.y) + 1, round(b.y), 1):
		# generate new position
		point = Vect3(a.x + ((b.x - a.x) * (y - a.y)) * inv_dist, y, a.z + ((b.z - a.z) * (y - a.y)) * inv_dist)
		tx_point = Vect2(ui.x + ((vi.x - ui.x) * (y - a.y)) * inv_dist, ui.y + ((vi.y - ui.y) * (y - a.y)) * inv_dist)
		line += ((point, tx_point,),)
	return line


def render(mesh, screen, depth, screen_write, textures, textures_index, far):
	""" This function handles rendering a mesh to the screen. """

	indices = [i for i in range(len(mesh))]
	indices = func.QuickSort(mesh.depth, indices)[1]

	for index in indices:
		face, texture = mesh[index], textures[textures_index[index]]
		txH, txW = texture.get_height() - 1, texture.get_width() - 1
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
					z = 1 / inv_z

					try:
						if depth[point[1]][point[0]] > inv_z:
							# Get current texture coordinate:
							dither = DITHER[y % 4][x % 4]
							uv = getUV(tls, trs, ls, rs, z, step)
							uv = int(uv[0] * txW + dither), int(uv[1] * txH + dither)
							colour = texture.get_at(uv)

							# Write to screen:
							screen.set_at(point, colour)
							screen_write.set_at(point, 1)
							depth[point[1]][point[0]] = inv_z
					except IndexError:
						pass


def renderSkybox(mesh, screen, screen_write, texture):
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
					try:
						if not bool(screen_write.get_at(point)):
							z = 1 / quickInterp(ls.z, ls.x, rs.z, inv_dist, step)
							dither = DITHER[y % 4][x % 4]
							uv = getUV(tls, trs, ls, rs, z, step)
							uv = int(uv[0] * txW + dither), int(uv[1] * txH + dither)
							colour = texture.get_at(uv)
							screen.set_at(point, colour)
					except IndexError:
						pass

