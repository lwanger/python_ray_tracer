"""
Ray tracer code -- command line version and basic routines

Len Wanger, copyright 2020
"""

# import numpy as np
from random import random
from geometry_classes import Vec3
from material_classes import ray_color


def render_chunk(world: "Scene", camera: "Camera", fb: "FrameBuffer", x_size: int, y_size: int, l:int, b:int,
                 r: int, t: int, samples_per_pixel: int, max_depth: int) -> "np.array":
    # TODO: make a numpy array (replace fb) and return
    use_r = min(r, x_size)
    use_t = min(t, y_size)
    # print(f'l={l}, r={r}, b={b}, t={t}, use_r={use_r}, use_t={use_t}')

    for j in range(b, use_t):
        for i in range(l, use_r):
            pixel_color = Vec3(0, 0, 0)

            for s in range(samples_per_pixel):
                # if self.render_cancelled is True:
                #     raise RenderCanceledException

                u = (i + random()) / (x_size - 1)
                v = (j + random()) / (y_size - 1)
                ray = camera.get_ray(u, v)
                pixel_color += ray_color(ray, world, max_depth)

            fb.set_pixel(i, j, pixel_color.get_unscaled_color(), samples_per_pixel)