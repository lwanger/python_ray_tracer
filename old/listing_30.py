"""
Simple camera

Listing 24 from Pete Shirley's Ray Tracing in a Weekend:

https://raytracing.github.io/books/RayTracingInOneWeekend.html

Add anti-aliasing

Len Wanger -- 2020
"""

import math
from random import random, uniform

import numpy as np

from tqdm import tqdm

from framebuffer import FrameBuffer, save_image, show_image
from geometry_classes import Vec3, Ray, Camera, Geometry, GeometryList, Sphere


ASPECT_RATIO = 16.0/9.0
X_SIZE = 384
Y_SIZE = int(X_SIZE/ASPECT_RATIO)
SAMPLES_PER_PIXEL = 5  # 100 in Shirley's code... very slow


def ray_color(ray: Ray, world: Geometry):
    hr = world.hit(ray, 0, math.inf)

    if hr is not None:
        return 0.5 * (hr.normal + Vec3(1,1,1))

    unit_direction = Vec3(ray.direction).unit_vector()
    t = 0.5 * (unit_direction.y + 1.0)
    return (1-t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0)


fb = FrameBuffer(X_SIZE, Y_SIZE, np.int8, 'rgb')
camera = Camera()
world = GeometryList()
world.add( Sphere(Vec3(0,0,-1), 0.5))
world.add( Sphere(Vec3(0,-100.5,-1), 100))

# write to framebuffer
for j in tqdm(range(Y_SIZE), desc="scanlines"):
    for i in range(X_SIZE):
        pixel_color = Vec3(0, 0, 0)

        for s in range(SAMPLES_PER_PIXEL):
            u = (i + random()) / (X_SIZE-1)
            v = (j + random()) / (Y_SIZE-1)
            ray = camera.get_ray(u, v)
            pixel_color += ray_color(ray, world)

        fb.set_pixel(i, j, pixel_color, SAMPLES_PER_PIXEL)

img = fb.make_image()
show_image(img)
save_image(img, "listing_9.png")
