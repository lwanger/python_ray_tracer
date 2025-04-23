"""
Test the framebuffe with an RGB image.

Listing 7 from Pete Shirley's Ray Tracing in a Weekend:

https://raytracing.github.io/books/RayTracingInOneWeekend.html#outputanimage/addingaprogressindicator

Len Wanger -- 2020
"""

import numpy as np
from PIL import Image

from framebuffer import FrameBuffer, save_image, show_image
from geometry_classes import Vec3

# X_SIZE = 512
# Y_SIZE = 512
#
# # test framebuffer utilities
# fb = FrameBuffer(512, 512, np.uint8, 'rgb')
#
# # write to framebuffer
# for y in range(Y_SIZE):
#     for x in range(X_SIZE):
#         r = x / (X_SIZE-1)
#         g = y / (X_SIZE-1)
#         b = 0.25
#         fb.set_pixel(x, y, [r,g,b])

X_SIZE = 200
Y_SIZE = 100

# test framebuffer utilities
fb = FrameBuffer(X_SIZE, Y_SIZE, np.uint8, 'rgb')

# write to framebuffer
for y in range(Y_SIZE):
    for x in range(X_SIZE):
        r = x / X_SIZE
        g = y / Y_SIZE
        b = 0.2
        ir = int(255.99*r)
        ig = int(255.99*g)
        ib = int(255.99*b)
        fb.set_pixel(x, y, [ir,ig,ib])

# show framebuffer
img = fb.make_image()
show_image(img)
save_image(img, "framebuffer_test2.png")
