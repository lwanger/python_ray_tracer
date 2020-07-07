"""
Simple camera

Listing 33 from Pete Shirley's Ray Tracing in a Weekend:

https://raytracing.github.io/books/RayTracingInOneWeekend.html

Add diffuse material and metal materials

Len Wanger -- 2020
"""

import math
from random import random

import numpy as np

from tqdm import tqdm

from framebuffer import FrameBuffer, save_image, show_image
from geometry_classes import Vec3, Ray, Camera, Geometry, GeometryList, Sphere
from geometry_classes import Lambertian, Metal


ASPECT_RATIO = 16.0/9.0
X_SIZE = 384
Y_SIZE = int(X_SIZE/ASPECT_RATIO)
SAMPLES_PER_PIXEL = 5  # 100 in Shirley's code... very slow. 5 is tolerable
MAX_DEPTH = 25 #  50


def ray_color(ray: Ray, world: Geometry, depth=1):
    if depth < 1:
        return Vec3(0, 0, 0)

    hr = world.hit(ray, 0.001, math.inf)

    if hr is not None:
        matl_record = hr.material.scatter(ray, hr)
        if matl_record.more:
            return matl_record.attenuation * ray_color(matl_record.scattered, world, depth-1)
        else:
            return Vec3(0,0,0)

    unit_direction = Vec3(ray.direction).unit_vector()
    t = 0.5 * (unit_direction.y + 1.0)
    return (1-t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0)


fb = FrameBuffer(X_SIZE, Y_SIZE, np.int8, 'rgb')
camera = Camera()

diffuse_1 = Lambertian(Vec3(0.7, 0.3, 0.3))
diffuse_2 = Lambertian(Vec3(0.8, 0.8, 0))
metal_1 = Metal(Vec3(0.8,0.6,0.2), fuzziness=0.3)
metal_2 = Metal(Vec3(0.8,0.6,0.8), fuzziness=1.0)

world = GeometryList()
world.add(Sphere(Vec3(0,0,-1), 0.5, diffuse_1))
world.add(Sphere(Vec3(0,-100.5,-1), 100, diffuse_2))
world.add(Sphere(Vec3(1,0,-1), 0.5, metal_1))
world.add(Sphere(Vec3(-1,0,-1),0.5, metal_2))

# write to framebuffer
for j in tqdm(range(Y_SIZE), desc="scanlines"):
    for i in range(X_SIZE):
        pixel_color = Vec3(0, 0, 0)

        for s in range(SAMPLES_PER_PIXEL):
            u = (i + random()) / (X_SIZE-1)
            v = (j + random()) / (Y_SIZE-1)
            ray = camera.get_ray(u, v)
            pixel_color += ray_color(ray, world, MAX_DEPTH)

        fb.set_pixel(i, j, pixel_color, SAMPLES_PER_PIXEL)

img = fb.make_image()
show_image(img)
save_image(img, "listing_9.png")
