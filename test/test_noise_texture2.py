"""
Test function to examine noise textures. Creates an image with a noise map for a region (ll,ur) and
displays summary statistics. Useful to test parameters for NoiseTexture's

Len Wanger, copyright 2020
"""

from pathlib import Path
import numpy as np

from framebuffer import FrameBuffer
from geometry_classes import get_color
# from vec3b import Vec3
from texture_classes2 import NoiseTexture
from perlin2 import value_noise, fractal_noise, turbulent_noise, marble_pattern, wood_pattern

import colorcet


def calc_array_stats(map, name, show_parms=False):
    # print(f'\tCalculating statistics for {name}')
    min = np.amin(map, axis=None)
    max = np.max(map)
    mean = np.mean(map)
    stddev = np.std(map)
    print(f'stats for {name}: min={min:0.4f}, max={max:0.4f}, mean={mean:0.4f}, stddev={stddev:0.4f}')
    if show_parms is True:
        print(f'\nUse parms: xlate={(-min):0.4f},  scale={(1/(max-min)):0.4f}')


def make_image(z: float=0.1, calc_stats=False):
    # p = Vec3(0,0,0)
    # p = (0, 0, 0.1)
    # print(f'make_image: z={z}')

    for j in range(im_height):
        for i in range(im_width):
            p = (i, j, z)

            if calc_stats is True:
                val = noise.raw_value(0, 0, p)
                raw_noise_map[j, i] = val

            val = noise.value(0, 0, p)
            fb.set_pixel(i, j, (val[0], val[1], val[2]))

    if calc_stats is True:
        calc_array_stats(raw_noise_map, 'raw_noise_map', True)

    im = fb.make_image()
    return im


if __name__ == '__main__':
    print('Noise texture testers:')

    # fire, jade, dimgray, coolwarm,; cc.kbc,; cc.blues  # clouds?, cc.rainbow, cc.  # wood? use part of range
    use_palette = 'jade2'

    palette_parms = {
        'fire': (colorcet.fire, 'fire'),
        'jade': (colorcet.kgy, 'jade'),
        'jade2': (colorcet.CET_D13[135:240], 'jade2'),
        'dimgray': (colorcet.dimgray, 'dimgray'),
        'wood': (colorcet.CET_CBC1[100:145], 'wood'),
        'wood2': (colorcet.CET_CBD1[190:], 'wood2'),
        'wood3': (colorcet.CET_D6[155:240], 'wood3'),
    }

    point_scale = 1.0

    palette, name = palette_parms[use_palette]
    colormap = [get_color(i,palette) for i in range(len(palette))]

    # use_eval: value_noise, value_noise2, fractal_noise, turbulent_noise, marble_pattern, wood_pattern
    use_eval = 'marble_pattern'

    if use_eval == 'value_noise':
        eval_func = value_noise
        kwargs = {'frequency': 0.5 }
        translate = 0.060
        scale = 8.0
    elif use_eval == 'value_noise2':
        eval_func = value_noise
        kwargs = {'frequency': 5.0}
        translate = 0.6457  # for freq = 5.0
        scale = 0.7682  # for freq = 5.0
    elif use_eval == 'fractal_noise':
        point_scale = 3.0
        eval_func = fractal_noise
        kwargs = {'frequency': 0.02, 'frequency_mult': 1.8, 'amplitude_mult': 0.35, 'layers': 5, 'div_val': 1.0 }
        translate = -0.233
        scale = 1.683
    elif use_eval == 'turbulent_noise':
        eval_func = turbulent_noise
        kwargs = {'frequency': 0.02, 'frequency_mult': 1.8, 'amplitude_mult': 0.35, 'layers': 5, 'div_val': 1.0}
        translate = -0.7578
        scale = 1.0308
    elif use_eval == 'marble_pattern':
        point_scale = 4.0  # delete me

        eval_func = marble_pattern
        # kwargs = {'frequency': 0.02, 'frequency_mult': 1.8, 'amplitude_mult': 0.35, 'layers': 5, 'displace_x': 100}
        # kwargs = {'frequency': 0.024, 'frequency_mult': 2.5, 'amplitude_mult': 0.35, 'layers': 7, 'displace_x': 200}
        kwargs = {'frequency': 0.024, 'frequency_mult': 2.5, 'amplitude_mult': 0.5, 'layers': 7, 'displace_x': 200}
        translate = 0.20
        scale = 1.0
    else:  # use_eval == 'wood_pattern':
        eval_func = wood_pattern
        kwargs = {'frequency': 0.01, 'frequency_mult': 10,}
        translate = 1.0
        scale = 0.5

    noise = NoiseTexture(colormap, point_scale=point_scale, translate=translate, scale=scale, name=name,
                            eval_func=eval_func, eval_kwargs=kwargs)

    # im_width = im_height = 512
    # im_width = im_height = 256
    im_width = im_height = 128
    # im_width = im_height = 32

    fb = FrameBuffer(x_size=im_width, y_size=im_height, depth="rgb")

    # generate value noise
    print(f'\tCreating noise_map  --  translate={translate}, scale={scale}\n')

    noise_map = np.zeros((im_width, im_height), dtype=float)
    raw_noise_map = np.zeros((im_width, im_height), dtype=float)

    # if True:  # single image
    if False:
        im = make_image(z=0.1, calc_stats=False)

        # im.show()
        p = Path('./perlin_test.gif')
        im.save(p)
    else:  # animated GIF
        # frames = im_width
        frames = im_width // 10
        print(f'creating animated GIF ({frames} frames):')
        images = []
        for z in range(frames):
            print(f'\tmaking frame {z}')
            im = make_image(z, calc_stats=False)
            # im = make_image(z/10, calc_stats=False)
            images.append(im)

        images[0].save('perlin_anim.gif', save_all=True, append_images=images[1:], duration=100, loop=0)