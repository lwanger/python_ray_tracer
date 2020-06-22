"""
Simple camera

Listing 60 from Pete Shirley's Ray Tracing in a Weekend:

https://raytracing.github.io/books/RayTracingInOneWeekend.html

Add movable camera

Len Wanger -- 2020
"""

import math
import os
from random import random, uniform

import numpy as np

from tqdm import tqdm

from framebuffer import FrameBuffer, save_image, show_image
from geometry_classes import Vec3, Ray, Camera, Geometry, GeometryList, Sphere
from material_classes import Lambertian, Metal, Dielectric


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

    unit_direction = ray.direction.unit_vector()
    t = 0.5 * (unit_direction.y + 1.0)
    return Vec3(1.0, 1.0, 1.0)*(1-t) + Vec3(0.5, 0.7, 1.0)*t

def create_simple_world():
    diffuse_1 = Lambertian(Vec3(0.7, 0.3, 0.3))
    diffuse_2 = Lambertian(Vec3(0.8, 0.8, 0))
    metal_1 = Metal(Vec3(0.8,0.6,0.2), fuzziness=0.3)
    dielectric_1 = Dielectric(1.5)

    world = GeometryList()
    world.add(Sphere(Vec3(0,0,-1), 0.5, diffuse_1))
    world.add(Sphere(Vec3(0,-100.5,-1), 100, diffuse_2))
    world.add(Sphere(Vec3(1,0,-1), 0.5, metal_1))
    world.add(Sphere(Vec3(-1,0,-1),0.5, dielectric_1))
    world.add(Sphere(Vec3(-1,0,-1),-0.45, dielectric_1))  # hollow sphere
    return world


def create_random_world():
    world = GeometryList()
    ground_material = Lambertian(Vec3(0.5,0.5,0.5))
    glass_material = Dielectric(1.5)
    center_offset = Vec3(4, 0.2, 9)

    world.add(Sphere(Vec3(0,-1000,0), 1000, ground_material))

    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random()
            center = Vec3(a+0.9*random(), 0.2, b+0.9*random())

            if (center - center_offset).length() > 0.9:
                if choose_mat < 0.8:  # diffuse
                    r = random()*random()
                    g = random()*random()
                    b = random()*random()
                    albedo = Vec3(r,g,b)
                    sphere_material = Lambertian(albedo)
                elif choose_mat < 0.95:  # metal
                    albedo = uniform(0.5, 1.0)
                    fuzz = uniform(0.0, 0.5)
                    sphere_material = Metal(Vec3(albedo, albedo, albedo), fuzz)
                else:  # glass
                    sphere_material = glass_material

                world.add(Sphere(center, 0.2, sphere_material))

    material_1 = Dielectric(1.5)
    world.add(Sphere(Vec3(0,1,0), 1.0, material_1))
    material_2 = Lambertian(Vec3(0.4, 0.2, 0.1))
    world.add(Sphere(Vec3(-4, 1, 0), 1.0, material_2))
    material_3 = Metal(Vec3(0.7,0.6,0.5), 0.0)
    world.add(Sphere(Vec3(4, 1, 0), 1.0, material_3))
    return world


if __name__ == '__main__':
    use_res = int(os.getenv("USE_RES", 2))
    image_file = os.getenv("IMAGE_FILE", "image.png")

    ASPECT_RATIO = 16.0 / 9.0

    if use_res == 0:  # ultra low res for debugging
        X_SIZE = 100
        SAMPLES_PER_PIXEL = 1
        MAX_DEPTH = 5
    elif use_res == 1:  # ultra low res for debugging
        X_SIZE = 200
        SAMPLES_PER_PIXEL = 4
        MAX_DEPTH = 10
    elif use_res == 2:  # normal res
        X_SIZE = 384
        SAMPLES_PER_PIXEL = 5
        MAX_DEPTH = 25
    else:  # ultra high res
        X_SIZE = 1024
        SAMPLES_PER_PIXEL = 100
        MAX_DEPTH = 50

    Y_SIZE = int(X_SIZE / ASPECT_RATIO)

    print(
        f'image_file={image_file}, use_res={use_res}: x_size={X_SIZE}, y_size={Y_SIZE}, samples_per_pixel={SAMPLES_PER_PIXEL}, max_depth={MAX_DEPTH}')

    fb = FrameBuffer(X_SIZE, Y_SIZE, np.int8, 'rgb')

    look_from = Vec3(13,2,3)
    look_at = Vec3(0,0,0)
    vup = Vec3(0,1,0)
    fd = 10.0
    aperature = 0.1
    camera = Camera(look_from, look_at, vup, 20, aperature=aperature, focus_dist=fd)

    world = create_random_world()

    # write to framebuffer
    for j in tqdm(range(Y_SIZE), desc="scanlines"):
        for i in range(X_SIZE):
            pixel_color = Vec3(0, 0, 0)

            for s in range(SAMPLES_PER_PIXEL):
                u = (i + random()) / (X_SIZE-1)
                v = (j + random()) / (Y_SIZE-1)
                ray = camera.get_ray(u, v)
                pixel_color += ray_color(ray, world, MAX_DEPTH)

            fb.set_pixel(i, j, pixel_color.get_unscaled_color(), SAMPLES_PER_PIXEL)

    img = fb.make_image()
    show_image(img)
    save_image(img, image_file)
