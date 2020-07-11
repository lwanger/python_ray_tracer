"""
Perlin noise

TODO:
    - color maps:
        parts of color maps (use brown region of CET_CBC1 and extrapolate to full 255)
        do green version of blues or kbc
    - cleaner code (shader language?) Have function to use (w/ or w/o normalization)
    - clean up -- two versions here...

In Progress:

from: https://mrl.nyu.edu/~perlin/noise/
2002 paper: https://mrl.nyu.edu/~perlin/paper445.pdf
more: https://mrl.nyu.edu/~perlin/homepage2006/bumpy/index.html

"""

# import functools
import math
from pathlib import Path
from random import random, randint, uniform, shuffle

import numpy as np

import colorcet as cc

from geometry_classes import get_color, clamp

import cooked_input as ci

from framebuffer import FrameBuffer
from geometry_classes import lerp, Vec3, dot

TWO_PI = 2*math.pi

K_MAX_TABLE_SIZE = 256
K_MAX_TABLE_SIZE_MASK = K_MAX_TABLE_SIZE - 1


def smoothstep(t: float) -> float:
    # cubic smoothing
    return t * t * (3 - 2 * t)


def fade(t: float) -> float:
    # quintic smoothing
    val = t * t * t * (t * (t * 6 - 15) + 10)
    return val


def stripes(x: float, f: float) -> float:
   t = .5 + .5 * math.sin(f * 2*math.pi * x)
   return t * t - .5


def fmt_list(lst):
    return f'[{", ".join((f"{v:0.2f}" for v in lst) )}]'


class ValueNoise3D():
    # Perlin noise
    def __init__(self):
        temp_table = []
        self.r = []

        # create array of random values and initialize permutaiton table
        for k in range(K_MAX_TABLE_SIZE):
            # add a random vector
            x = uniform(-1,1)
            y = uniform(-1,1)
            z = uniform(-1,1)
            self.r.append(Vec3(x,y,z))
            temp_table.append(k)

        # shuffle values to make permutation tables
        shuffle(temp_table)

        self.perm_x_table = temp_table + temp_table
        shuffle(temp_table)
        self.perm_y_table = temp_table + temp_table
        shuffle(temp_table)
        self.perm_z_table = temp_table + temp_table


    def __repr__(self):
        rs = [repr(vec) for vec in self.r[:5]]
        r_s = f'[{", ".join(rs)}]'
        px_s = fmt_list(self.perm_x_table[:5])
        py_s = fmt_list(self.perm_y_table[:5])
        pz_s = fmt_list(self.perm_z_table[:5])
        # return f'r[:5]={fmt_list(self.r[:5])}, perm_x[:5]={fmt_list(self.perm_x_table[:5])}, perm_y[:5]={fmt_list(self.perm_y_table[:5])}, perm_z[:5]={fmt_list(self.perm_z_table[:5])}'
        return f'r[:5]={r_s}, perm_x[:5]={px_s}, perm_y[:5]={py_s}, perm_z[:5]={pz_s}'


    def perlin_interp(self, c: Vec3, u: float, v: float, w: float) -> float:
        accum = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    weight_v = Vec3(u-i, v-j, w-k)
                    weight = dot(c[i * 4 + j * 2 + k], weight_v)
                    accum += (i*u + (1-i)*(1-u)) * (j*v + (1-j)*(1-v)) * (k*w + (1-k)*(1-w)) * weight

        return accum


    def eval(self, x, y, z):
        i = math.floor(x)
        j = math.floor(y)
        k = math.floor(z)
        u = x - i
        v = y - j
        w = z - k

        su = smoothstep(u)
        sv = smoothstep(v)
        sw = smoothstep(w)

        c = [0] * 8
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    idx = self.perm_x_table[(i + di) & 255] ^ self.perm_y_table[(j + dj) & 255] ^ self.perm_z_table[(k + dk) & 255]
                    val = self.r[idx]
                    idx = di * 4 + dj * 2 + dk
                    c[idx] = val

        return self.perlin_interp(c, su, sv, sw)


def lumpy_noise(noise, px, py, scale=0.03, lumpiness=8):
    # LUMPY: 	 .03 * noise(8*x,8*y,8*z)
    val = scale * noise.eval(lumpiness*px, lumpiness*py)
    return val

def crinkly_noise(noise, px, py, scale=-0.1, frequency=0.02, frequencyMult=1.8, amplitudeMult=0.35,
                  numLayers=5, maxNoiseVal=0):
    # CRINKLY: 	-.10 * turbulence(x,y,z)
    val = scale * turbulent_noise(noise, i, j, frequency, frequencyMult, amplitudeMult, layers=5)
    return val


def marble_pattern(noise, x, y, frequency, frequency_mult, amplitude_mult, layers=5):
    px = x * frequency
    py = y * frequency
    amplitude = 1
    val = 0

    for l in range(layers):
        val +=  noise.eval(px, py) * amplitude
        px *= frequency_mult
        py *= frequency_mult
        amplitude *= amplitude_mult

    # displace the value i used in the sin expression by noisevValue * 100
    ret_val = (math.sin((x + val * 100) * TWO_PI / 200) + 1) / 2.0

    return ret_val


def marble_pattern2(noise, x, y, scale=0.01, frequency=0.02, frequencyMult=1.8, amplitudeMult=0.35,
                  numLayers=5, maxNoiseVal=0):
    #MARBLED: 	 .01 * stripes(x + 2*turbulence(x,y,z), 1.6);
    t_val = turbulent_noise(noise, i, j, frequency, frequencyMult, amplitudeMult, layers=5)
    val = scale * stripes(x + 2*t_val, 1.6)
    return val


def wood_pattern(noise, x, y, frequency=0.01):
    g = noise.eval(x*frequency, y*frequency) * 10
    val = g - int(g)
    return val

###
# noise functions to pass in to NoiseTexture eval-func
###
# def value_noise(noise, i, j, k, frequency=1.0, translate=0.0, scale=1.0):
def value_noise(noise, i, j, k, frequency=1.0):
    val = noise.eval(i, j, k) * frequency
    # val = (val + translate) * scale  # shift and scale to 0.0-1.0
    # val = clamp(val, 0.0, 1.0)  # clamp to 0.0-1.0
    return val
    # return noise.eval(i, j, k) * frequency


def turbulent_noise(noise, i, j, k, frequency, frequency_mult, amplitude_mult, layers=5, sub_val=0.0, div_val=1.0):
    pi = i * frequency
    pj = j * frequency
    pk = k * frequency
    amplitude = 1
    val = 0

    for l in range(layers):
        # val += abs((2 * noise.eval(pi, pj) - 1) * amplitude)
        val += abs((2 * noise.eval(pi, pj, pk) - 1) * amplitude)
        pi *= frequency_mult
        pj *= frequency_mult
        amplitude *= amplitude_mult

    # val = (val-sub_val) / div_val
    scaled_val = val / div_val
    return (0.5 * (1.0 + scaled_val))
    # return val


def fractal_noise(noise, i, j, k, frequency, frequency_mult, amplitude_mult, layers=5, sub_val=0.0, div_val=1.0):
    pi = i * frequency
    pj = j * frequency
    pk = k * frequency
    amplitude = 1
    val = 0

    for l in range(layers):
        val += noise.eval(pi, pj, pk) * amplitude
        pi *= frequency_mult
        pj *= frequency_mult
        pk *= frequency_mult
        amplitude *= amplitude_mult

    # val = (val-sub_val) / div_val
    scaled_val = val / div_val
    return (0.5 * (1.0 + scaled_val))
    # return val


if __name__ == '__main__':
    # palettes: see: https://colorcet.holoviz.org/user_guide/index.html
    # examples: fire, colorwheel, bkr, bky, bwy coolwarm isolum gray dim_gray cwr kgy kb kg kr kbc blues rainbow
    # palette = cc.fire
    # palette = cc.coolwarm
    palette = cc.dimgray
    # palette = cc.kgy  # jade
    # palette = cc.kbc
    # palette = cc.blues  # clouds?
    # palette = cc.rainbow
    # palette = cc.CET_CBC1  # wood? use part of range
    colormap = [get_color(i,palette) for i in range(len(palette))]
    # im_width = 512
    im_width = 32
    # im_height = 512
    im_height = 32
    # fb = FrameBuffer(x_size=im_width, y_size=im_height)  # depth = "s" for monochrome
    fb = FrameBuffer(x_size=im_width, y_size=im_height, depth="rgb")

    # generate value noise
    noise_map = np.zeros((im_width, im_height), dtype=float)

    noise = ValueNoise3D()

    # if False:  # value noise
    if True:  # value noise
        freq = 0.5

        for j in range(im_height):
            for i in range(im_width):
                # val = noise.eval(i,j) * freq
                val = value_noise(noise, i+0.1, j+0.1, 0.1, freq)
                # val = noise.raw_noise(0, 0, Vec3(i+0.1, j+0.1, 0.1)) #  () value_noise(noise, i+0.1, j+0.1, 0.1, freq)

                val = (val + 0.05) * 10  # min is -0.06 - 0.06 map to 0.0-1.0
                val = clamp(val, 0.0, 1.0)

                fb.set_pixel(i, j, colormap[int(val * 255.999)])
                noise_map[j][i] = val

        print('\tCalculating statistics')
        min = np.amin(noise_map, axis=None)
        max = np.max(noise_map)
        mean = np.mean(noise_map)
        stddev = np.std(noise_map)
        print(f'stats: min={min:0.4f}, max={max:0.4f}, mean={mean:0.4f}, stddev={stddev:0.4f}')

    elif True: # fractal noise
    # elif False: # fractal noise
        frequency = 0.02
        frequencyMult = 1.8
        amplitudeMult = 0.35
        numLayers = 5
        maxNoiseVal = 0
        minNoiseVal = 1000000

        # fractal min=-0.5589, max=.52269
        # turbuent min=0.458, max=2.741
        # sub_val = 0.458
        sub_val = 0
        # div_val = 2.742-sub_val
        div_val = 1.0

        for j in range(im_height):
            for i in range(im_width):
                val = fractal_noise(noise, i, j, 0, frequency, frequencyMult, amplitudeMult, layers=5,
                                    sub_val=sub_val, div_val=div_val)
                # val = turbulent_noise(noise, i, j, 0, frequency, frequencyMult, amplitudeMult,
                #                       layers=5, sub_val=sub_val, div_val=div_val)
                # val = marble_pattern(noise, i, j, frequency, frequencyMult, amplitudeMult, layers=5)
                # val = lumpy_noise(noise, i, j)
                # val = crinkly_noise(noise, i, j, scale=-0.1)  #  needs debugging
                # val = marble_pattern2(noise, i, j)
                noise_map[j,i] = val
                maxNoiseVal = max(val, maxNoiseVal)
                minNoiseVal = min(val, minNoiseVal)

        print(f'minNoiseVal={minNoiseVal}, maxNoiseVal={maxNoiseVal}')

        for j in range(im_height):
            for i in range(im_width):
                val = noise_map[j,i]
                scaled_val = val / maxNoiseVal
                # fb.set_pixel(i, j, int(scaled_val * 255.999))  # monochrome with frambuffer depth "s"
                # fb.set_pixel(i, j, colormap[int(scaled_val * 255.999)])

                pv = (0.5 * (1.0 + scaled_val))
                fb.set_pixel(i, j, colormap[int(pv * 255.999)])  # ValueNoise3D can return negative, so...
    else:  # wood
        for j in range(im_height):
            for i in range(im_width):
                val = wood_pattern(noise, i, j, frequency=0.01)
                # fb.set_pixel(i, j, int(val * 255.999))  # monochrome with frambuffer depth "s"
                fb.set_pixel(i, j, colormap[int(val * 255.999)])



    im = fb.make_image()
    im.show()
    if ci.get_yes_no(prompt='save image? ', default="no") == "yes":
        fname = ci.get_string(prompt='filename ', default='perlin.png')
        p = Path(fname)
        # with open(p, "w") as f:
        #     im.save(f)
        im.save(p)

