"""
Ray tracer code -- command line version and basic routines

Len Wanger, copyright 2020
"""

from datetime import datetime
import os
# from random import random

import numpy as np

from tqdm import tqdm

from framebuffer import FrameBuffer, save_image
from material_classes import ray_color
from create_scene_funcs import *


#CREATOR_FUNC = create_canonical_1
CREATOR_FUNC = create_perlin_1


res_settings = {
        'profile': { 'x_size': 50, 'chunk_size': 10, 'samples_per_pixel': 25, 'samples_per_light': 20, 'max_depth': 20 },
        'low': { 'x_size': 100, 'chunk_size': 10, 'samples_per_pixel': 10, 'samples_per_light': 10, 'max_depth': 10 },
        'med': { 'x_size': 200, 'chunk_size': 25, 'samples_per_pixel': 25, 'samples_per_light': 20, 'max_depth': 20 },
        'high': { 'x_size': 384, 'chunk_size': 25, 'samples_per_pixel': 50, 'samples_per_light': 50, 'max_depth': 25 },
        'ultra': { 'x_size': 1024, 'chunk_size': 25, 'samples_per_pixel': 100, 'samples_per_light': 100, 'max_depth': 50 },
}


def env_or_defaults(var_name, def_val):
    try:
        return os.environ[var_name]
    except KeyError:
        return def_val

def get_render_settings():
    use_res = env_or_defaults('USE_RES', 'med').lower()
    settings = res_settings[use_res]
    
    x_size = int(env_or_defaults('X_SIZE', settings['x_size']))
    settings['x_size'] = x_size
    aspect_ratio = eval(env_or_defaults('ASPECT_RATIO', '16/9'))
    settings['aspect_ratio'] = aspect_ratio
    settings['chunk_size'] = int(env_or_defaults('CHUNK_SIZE', settings['chunk_size']))
    settings['samples_per_pixel'] = int(env_or_defaults('SAMPLES_PER_PIXEL', settings['samples_per_pixel']))
    settings['samples_per_light'] = int(env_or_defaults('SAMPLES_PER_LIGHT', settings['samples_per_light']))
    settings['max_depth'] = int(env_or_defaults('MAX_DEPTH', settings['max_depth']))
    settings['image_filename'] = env_or_defaults('IMAGE_FILENAME', 'rt.png')

    if env_or_defaults('RANDOM_CHUNKS', True) in {'0', 'False', 'FALSE', 'false', 'no', 'No', 'NO'}:
        settings['random_chunks'] = False
    else:
        settings['random_chunks'] = True

    settings['y_size'] = int(x_size / aspect_ratio)

    viewport_height = 2.0
    viewport_width = aspect_ratio * viewport_height
    focal_length = 1.0

    origin = Vec3(0.0, 0.0, 0.0)
    settings['origin'] = Vec3(0.0, 0.0, 0.0)
    horizontal = Vec3(viewport_width, 0, 0)
    settings['horizontal'] = Vec3(viewport_width, 0, 0)
    vertical = Vec3(0, viewport_height, 0)
    settings['vertical'] = Vec3(0, viewport_height, 0)
    settings['lower_left'] = origin - horizontal.div_val(2) - vertical.div_val(2) - Vec3(0, 0, focal_length)
        
    return settings


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

            
def render(world, camera, fb, x_size, y_size, samples_per_pixel, max_depth,  chunk_size=25):
    # simple version to render for command line
    start_time = datetime.now()
    x_chunks, r = divmod(x_size, chunk_size)
    
    if r != 0:
        x_chunks += 1

    y_chunks, r = divmod(y_size, chunk_size)
    if r != 0:
        y_chunks += 1

    # add tqmd
    total_chunks = x_chunks * y_chunks
    chunk_num = 1
    tq = tqdm(total=total_chunks, unit='chunks')

    for j in range(y_chunks):
        for i in range(x_chunks):
            l = i*chunk_size
            r = l + chunk_size
            b = j*chunk_size
            t = b + chunk_size
            render_chunk(world, camera, fb, x_size, y_size, l, b, r, t, samples_per_pixel, max_depth)
            # self.update_canvas(l, b, chunk_num, total_chunks)
            chunk_num += 1
            tq.update(n=chunk_num)

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    # finish_render(elapsed_time)
    return elapsed_time


if __name__ == '__main__':
    import dotenv

    dotenv.load_dotenv()
    settings = get_render_settings()
    world = CREATOR_FUNC()
    scene = world['scene']
    camera = world['camera']
    fb = FrameBuffer(settings['x_size'], settings['y_size'], np.int8, 'rgb')

    x_size = settings['x_size']
    y_size = settings['y_size']
    samples_per_pixel = settings['samples_per_pixel']
    max_depth = settings['max_depth']
    chunk_size = settings['chunk_size']
    image_filename = settings['image_filename']

    print(f'starting render')
    elapsed_time = render(scene, camera, fb, x_size, y_size, samples_per_pixel, max_depth, chunk_size)

    img = fb.make_image()
    save_image(img, image_filename)
    print(f'\nimage saved ({image_filename}) -- total time = {elapsed_time}')
