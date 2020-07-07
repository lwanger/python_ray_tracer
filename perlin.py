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

LUMPY: 	 .03 * noise(8*x,8*y,8*z);
CRINKLY: 	-.10 * turbulence(x,y,z);
MARBLED: 	 .01 * stripes(x + 2*turbulence(x,y,z), 1.6);

double stripes(double x, double f) {
   double t = .5 + .5 * Math.sin(f * 2*Math.PI * x);
   return t * t - .5;
}

double turbulence(double x, double y, double z) {
   double t = -.5;
   for (double f = 1 ; f <= W/12 ; f *= 2)
      t += Math.abs(noise(f*x,f*y,f*z) / f);
   return t;
}

public final class ImprovedNoise {
   static public double noise(double x, double y, double z) {
      int X = (int)Math.floor(x) & 255,                  // FIND UNIT CUBE THAT
          Y = (int)Math.floor(y) & 255,                  // CONTAINS POINT.
          Z = (int)Math.floor(z) & 255;
      x -= Math.floor(x);                                // FIND RELATIVE X,Y,Z
      y -= Math.floor(y);                                // OF POINT IN CUBE.
      z -= Math.floor(z);
      double u = fade(x),                                // COMPUTE FADE CURVES
             v = fade(y),                                // FOR EACH OF X,Y,Z.
             w = fade(z);
      int A = p[X  ]+Y, AA = p[A]+Z, AB = p[A+1]+Z,      // HASH COORDINATES OF
          B = p[X+1]+Y, BA = p[B]+Z, BB = p[B+1]+Z;      // THE 8 CUBE CORNERS,

      return lerp(w, lerp(v, lerp(u, grad(p[AA  ], x  , y  , z   ),  // AND ADD
                                     grad(p[BA  ], x-1, y  , z   )), // BLENDED
                             lerp(u, grad(p[AB  ], x  , y-1, z   ),  // RESULTS
                                     grad(p[BB  ], x-1, y-1, z   ))),// FROM  8
                     lerp(v, lerp(u, grad(p[AA+1], x  , y  , z-1 ),  // CORNERS
                                     grad(p[BA+1], x-1, y  , z-1 )), // OF CUBE
                             lerp(u, grad(p[AB+1], x  , y-1, z-1 ),
                                     grad(p[BB+1], x-1, y-1, z-1 ))));
   }
   static double fade(double t) { return t * t * t * (t * (t * 6 - 15) + 10); }
   static double lerp(double t, double a, double b) { return a + t * (b - a); }
   static double grad(int hash, double x, double y, double z) {
      int h = hash & 15;                      // CONVERT LO 4 BITS OF HASH CODE
      double u = h<8 ? x : y,                 // INTO 12 GRADIENT DIRECTIONS.
             v = h<4 ? y : h==12||h==14 ? x : z;
      return ((h&1) == 0 ? u : -u) + ((h&2) == 0 ? v : -v);
   }
   static final int p[] = new int[512], permutation[] = { 151,160,137,91,90,15,
   131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
   190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
   88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
   77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
   102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
   135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
   5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
   223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
   129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
   251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
   49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
   138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
   };
   static { for (int i=0; i < 256 ; i++) p[256+i] = p[i] = permutation[i]; }
}
"""

import functools
import math

import colorcet as cc

from geometry_classes import lerp


def smoothstep(t: float) -> float:
    return t * t * (3 - 2 * t)


def fade(t: float) -> float:
    val = t * t * t * (t * (t * 6 - 15) + 10)
    return val


def stripes(x: float, f: float) -> float:
   t = .5 + .5 * math.sin(f * 2*math.pi * x)
   return t * t - .5


def turbulence(x: float,  y: float,  z: float, W=512, use_abs=True) -> float:
    # turbulence is the sum of 1/f(noise) -- fractal sum
    #   = noise(p) + 1/2*(noise(2p) + 1/4*(noise(4p) + ...
    # until too small too see (W/12?)
    # use abs is for 1/f(|noise|) - used for marble
    t = -.5
    f = 1
    while True:
        # t += math.abs(noise(f * x, f * y, f * z) / f)
        if use_abs is True:
            t += math.abs(ImprovedNoise(f * x, f * y, f * z) / f)
        else:
            t += ImprovedNoise(f * x, f * y, f * z) / f

        # if f > W/12:
        if f > W/12:
            break
        f *= 2
    return t


def grad(hash: int, x: float, y: float, z: float) -> float:
    # convert low 4 bits of hash code into 12 gradient directions
    h = hash & 15
    if h < 8:
        u = x
    else:
        u = y

    if h < 4:
        v = y
    elif h==12 or h==14:
        v = x
    else:
        v = z

    if h & 1 == 1:
        u = -u

    if h & 2 == 1:
        v = -v

    return u + v


def ImprovedNoise(x: float, y: float, z: float) -> float:
    # find unit cube that contains point
    X = math.floor(x) & 255
    Y = math.floor(y) & 255
    Z = math.floor(z) & 255

    # fube relative x,y,z of point in cuve
    x -= math.floor(x)
    y -= math.floor(y)
    z -= math.floor(z)

    # compute fade curves for x, y, and z
    u = fade(x)
    v = fade(y)
    w = fade(z)

    # hash coordinates of the 8 cube corners
    A  = p[X] + Y
    AA = p[A] + Z
    AB = p[A+1] + Z

    B = p[X+1] + Y
    BA = p[B] + Z
    BB = p[B+1] + Z


    """
    lerp(w, lerp(v, lerp(u, grad(p[AA  ], x  , y  , z   ),  // AND ADD
                                     grad(p[BA  ], x-1, y  , z   )), // BLENDED
                             lerp(u, grad(p[AB  ], x  , y-1, z   ),  // RESULTS
                                     grad(p[BB  ], x-1, y-1, z   ))),// FROM  8
                     lerp(v, lerp(u, grad(p[AA+1], x  , y  , z-1 ),  // CORNERS
                                     grad(p[BA+1], x-1, y  , z-1 )), // OF CUBE
                             lerp(u, grad(p[AB+1], x  , y-1, z-1 ),
                                     grad(p[BB+1], x-1, y-1, z-1 ))))
    """
    lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z),
    grad(p[BA], x - 1, y, z)),
    lerp(u, grad(p[AB], x, y - 1, z),
    grad(p[BB], x - 1, y - 1, z))),
    lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1),
    grad(p[BA + 1], x - 1, y, z - 1)),
    lerp(u, grad(p[AB + 1], x, y - 1, z - 1),
         grad(p[BB + 1], x - 1, y - 1, z - 1))))


    lo = lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
        lerp(u, grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z)))

    hi = lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1), grad(p[BA + 1], x - 1, y, z - 1)),
        lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1)))

    return lerp(w, lo, hi)


permutation = [ 151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233,
    7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6,
    148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
    88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166,
    77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244,
    102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196,
    135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123,
    5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
    223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228,
    251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107,
    49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
    138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180
]

# static { for (int i=0; i < 256 ; i++) p[256+i] = p[i] = permutation[i]; }
# p = [permutation[i%256] for i in range(512)]
#
# print(p)

"""
https://www.scratchapixel.com/code.php?id=55&origin=/lessons/procedural-generation-virtual-worlds/procedural-patterns-noise-part-1
"""
from random import random, shuffle
K_MAX_TABLE_SIZE = 256
K_MAX_TABLE_SIZE_MASK = K_MAX_TABLE_SIZE - 1

class ValueNoise():
    def __init__(self, seed=2016):
        # TODO: random seed
        temp_table = []
        self.r = []

        # create array of random values and initialize permutaiton table
        for k in range(K_MAX_TABLE_SIZE):
            self.r.append( random() )
            temp_table.append(k)

        # shuffle values
        shuffle(temp_table)
        self.permutation_table = temp_table + temp_table

    def eval(self, x, y):
        # evaluate the noise function
        xi = math.floor(x)
        yi = math.floor(y)
 
        tx = x - xi
        ty = y - yi
 
        rx0 = xi & K_MAX_TABLE_SIZE_MASK
        rx1 = (rx0+1) & K_MAX_TABLE_SIZE_MASK
        ry0 = yi & K_MAX_TABLE_SIZE_MASK
        ry1 = (ry0+1) & K_MAX_TABLE_SIZE_MASK

        # random values at the corners of the cell using permutation table
        c00 = self.r[self.permutation_table[self.permutation_table[rx0] + ry0]]
        c10 = self.r[self.permutation_table[self.permutation_table[rx1] + ry0]]
        c01 = self.r[self.permutation_table[self.permutation_table[rx0] + ry1]]
        c11 = self.r[self.permutation_table[self.permutation_table[rx1] + ry1]]

        # remapping of tx and ty using the Smoothstep function
        # sx = smoothstep(tx)
        sx = fade(tx)
        # sy = smoothstep(ty)
        sy = fade(ty)

        # linearly interpolate values along the x axis
        nx0 = lerp(c00, c10, sx)
        nx1 = lerp(c01, c11, sx)
 
        # linearly interpolate the nx0/nx1 along they y axis
        return lerp(nx0, nx1, sy)



from framebuffer import FrameBuffer
import numpy as np

TWO_PI = 2*math.pi


def fractal_noise(noise, x, y, frequency, frequency_mult, amplitude_mult, layers=5):
    px = x * frequency
    py = y * frequency
    amplitude = 1
    val = 0

    for l in range(layers):
        val += noise.eval(px, py) * amplitude
        px *= frequency_mult
        py *= frequency_mult
        amplitude *= amplitude_mult

    return val


def turbulent_noise(noise, x, y, frequency, frequency_mult, amplitude_mult, layers=5):
    px = x * frequency
    py = y * frequency
    amplitude = 1
    val = 0

    for l in range(layers):
        val += abs((2 * noise.eval(px, py) - 1) * amplitude)
        px *= frequency_mult
        py *= frequency_mult
        amplitude *= amplitude_mult

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


def wood_pattern(noise, x, y, frequency=0.01):
    g = noise.eval(x*frequency, y*frequency) * 10
    val = g - int(g)
    return val


def hex_to_rgb(hex):
    # convert from hex string ("#FFFFFF") to rgb
    return ( int(hex[1:3], 16) / 255.999, int(hex[3:5], 16) / 255.999, int(hex[5:], 16) / 255.999 )


# @functools.lru_cache(maxsize=None)
def get_color(val, colormap=cc.m_fire):
    # use colorcet to get color value
    colormap_val = colormap[val]

    if isinstance(colormap_val, str):  # colormap is in hex
        color = hex_to_rgb(colormap_val)
        return color
    else:
        return colormap_val

if __name__ == '__main__':
    # palettes: see: https://colorcet.holoviz.org/user_guide/index.html
    # examples: fire, colorwheel, bkr, bky, bwy coolwarm isolum gray dim_gray cwr kgy kb kg kr kbc blues rainbow
    # palette = cc.fire
    # palette = cc.coolwarm
    # palette = cc.dimgray
    # palette = cc.kgy  # jade
    # palette = cc.kbc
    palette = cc.blues  # clouds?
    # palette = cc.rainbow
    # palette = cc.CET_CBC1  # wood? use part of range
    colormap = [get_color(i,palette) for i in range(len(palette))]
    im_width = 512
    im_height = 512
    # fb = FrameBuffer(x_size=im_width, y_size=im_height)  # depth = "s" for monochrome
    fb = FrameBuffer(x_size=im_width, y_size=im_height, depth="rgb")

    # generate value noise
    noise_map = np.zeros((im_width, im_height), dtype=float)
    noise = ValueNoise()

    if False:  # value noise
    # if True:  # value noise
        freq = 0.5

        for j in range(im_height):
            for i in range(im_width):
                val = noise.eval(i,j) * freq
                fb.set_pixel(i, j, colormap[int(val * 255.999)])

    elif True: # fractal noise
    # elif False: # fractal noise
        frequency = 0.02
        frequencyMult = 1.8
        amplitudeMult = 0.35
        numLayers = 5
        maxNoiseVal = 0

        for j in range(im_height):
            for i in range(im_width):
                val = fractal_noise(noise, i, j, frequency, frequencyMult, amplitudeMult, layers=5)
                # val = turbulent_noise(noise, i, j, frequency, frequencyMult, amplitudeMult, layers=5)
                # val = marble_pattern(noise, i, j, frequency, frequencyMult, amplitudeMult, layers=5)
                noise_map[j,i] = val
                maxNoiseVal = max(val, maxNoiseVal)

        for j in range(im_height):
            for i in range(im_width):
                val = noise_map[j,i]
                scaled_val = val / maxNoiseVal
                # fb.set_pixel(i, j, int(scaled_val * 255.999))  # monochrome with frambuffer depth "s"
                fb.set_pixel(i, j, colormap[int(scaled_val * 255.999)])
    else:  # wood
        for j in range(im_height):
            for i in range(im_width):
                val = wood_pattern(noise, i, j, frequency=0.01)
                # fb.set_pixel(i, j, int(val * 255.999))  # monochrome with frambuffer depth "s"
                fb.set_pixel(i, j, colormap[int(val * 255.999)])


    im = fb.make_image()
    im.show()

