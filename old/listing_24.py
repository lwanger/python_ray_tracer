"""
Simple camera

Listing 24 from Pete Shirley's Ray Tracing in a Weekend:

https://raytracing.github.io/books/RayTracingInOneWeekend.html

Add a hit list and geometry class.

Len Wanger -- 2020
"""

import math
import numpy as np

from tqdm import tqdm

from framebuffer import FrameBuffer, save_image, show_image
from geometry_classes import Vec3, Ray, dot, Geometry, GeometryList, Sphere


ASPECT_RATIO = 16.0/9.0
X_SIZE = 384
Y_SIZE = int(X_SIZE/ASPECT_RATIO)


def ray_color(ray: Ray, world: Geometry):
    hr = world.hit(ray, 0, math.inf)

    if hr is not None:
        return 0.5 * (hr.normal + Vec3(1,1,1))

    unit_direction = Vec3(ray.direction).unit_vector()
    t = 0.5 * (unit_direction.y + 1.0)
    return (1-t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0)


# test framebuffer utilities
fb = FrameBuffer(X_SIZE, Y_SIZE, np.int8, 'rgb')
print(f'fb shape: {fb.get_shape()}')

viewport_height = 2.0
viewport_width = ASPECT_RATIO * viewport_height
focal_length = 1.0

origin = Vec3(0.0, 0.0, 0.0)
horizontal = Vec3(viewport_width, 0, 0)
vertical = Vec3(0, viewport_height, 0)
lower_left = origin - horizontal/2 - vertical/2 - Vec3(0, 0, focal_length)

world = GeometryList()
world.add( Sphere(Vec3(0,0,-1), 0.5))
world.add( Sphere(Vec3(0,-100.5,-1), 100))

# write to framebuffer
for j in tqdm(range(Y_SIZE), desc="scanlines"):
    for i in range(X_SIZE):
        u = i / (X_SIZE-1)
        v = j / (Y_SIZE-1)
        direction = Vec3(lower_left + u*horizontal + v*vertical -origin)
        ray = Ray(origin, direction)
        color = ray_color(ray, world)
        fb.set_pixel(i,j,color)

# show framebuffer
img = fb.make_image()
show_image(img)
save_image(img, "listing_9.png")
