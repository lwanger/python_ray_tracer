"""
CS292 Class Assignment 1:

    https://www.youtube.com/watch?v=GuakuIWE278

Draw a (filled) box

1) main program to read coords (Xl, Yt, Xr, Yb) and intensity value

2) write to framebuffer -- and write framebuffer to std file fmt


"""

from pathlib import Path
import sys

import numpy as np
from PIL import Image

from framebuffer import FrameBuffer, save_image


X_SIZE = 200
Y_SIZE = 200


def draw_rect(fb: FrameBuffer, xl: int, yt: int, xr: int, yb: int, intensity):
    for y in range(yt, yb+1):
        for x in range(xl, xr+1):
            fb.set_pixel(x, y, intensity)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(f"{sys.argv[0]}: Xleft Ytop Xright Ybottom intensity_value")
        sys.exit(1)

    # Should really check the values... instead we'll just crash!
    xl = int(sys.argv[1])
    yt = int(sys.argv[2])
    xr = int(sys.argv[3])
    yb = int(sys.argv[4])
    intensity = int(sys.argv[5])

    fb = FrameBuffer(X_SIZE, Y_SIZE, np.int8)
    draw_rect(fb, xl, yt, xr, yb, intensity)

    img = fb.make_image()
    # show_image(img)
    img_name = Path(sys.argv[0]).stem + ".png"
    filename = Path("images", img_name)
    save_image(img, filename)
