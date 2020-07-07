"""
Simple camera

Listing 11 from Pete Shirley's Ray Tracing in a Weekend:

https://raytracing.github.io/books/RayTracingInOneWeekend.html

https://raytracing.github.io/books/RayTracingInOneWeekend.html#outputanimage/addingaprogressindicator

Improve sphere intersection check

Len Wanger -- 2020
"""

import math
import numpy as np

from framebuffer import FrameBuffer, save_image, show_image
from geometry_classes import Vec3, Ray, dot


ASPECT_RATIO = 16.0/9.0
X_SIZE = 384
Y_SIZE = int(X_SIZE/ASPECT_RATIO)


def hit_sphere(center: Vec3, radius: float, ray: Ray):
    oc = Vec3(ray.origin - center)
    a = ray.direction.squared_length()
    half_b = dot(oc, ray.direction)
    c = oc.squared_length() - radius**2
    discriminant = half_b**2 - a * c

    if discriminant < 0:
        return -1.0
    else:
        return (-half_b - math.sqrt(discriminant)) / a


def ray_color(ray: Ray):
    t = hit_sphere(Vec3(0,0,-1), 0.5, ray)
    if t > 0.0:
        n1 = Vec3(ray.at(t)) - Vec3(0,0,-1)
        n = n1.unit_vector()
        return 0.5 * Vec3(n.x+1, n.y+1, n.z+1)

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

from tqdm import tqdm

# write to framebuffer
for j in tqdm(range(Y_SIZE), desc="scanlines"):
    for i in range(X_SIZE):
        u = i / (X_SIZE-1)
        v = j / (Y_SIZE-1)
        direction = Vec3(lower_left + u*horizontal + v*vertical -origin)
        ray = Ray(origin, direction)
        color = ray_color(ray)
        fb.set_pixel(i,j,color)

# show framebuffer
img = fb.make_image()
show_image(img)
save_image(img, "listing_9.png")
