"""
Frame buffer utilities

Len Wanger -- 2020
"""

from math import sqrt
import numpy as np
from PIL import Image


ORIGIN_LL = 0
ORIGIN_UL = 1


class FrameBuffer():
    def set_pixel(self, x, y, value, samples=1):
        if samples == 1:
            scaled_value = value
        else:
            scale = 1 / samples
            # scaled_value = [sqrt(value.x*scale), sqrt(value.y*scale), sqrt(value.z*scale)]  # sqrt to gamma correct
            scaled_value = [sqrt(v*scale) for v in value]  # sqrt to gamma correct

        if self.origin == ORIGIN_LL:
            use_y = self.y_size - y - 1
        else:
            use_y = y

        if self.depth_num == 1:
            self.fb[use_y,x] = scaled_value
        elif self.depth_num == 3:
            if isinstance(scaled_value[0], float):  # convert to int 0..256
                self.fb[use_y, x, 0] = int(scaled_value[0] * 255.999)
                self.fb[use_y, x, 1] = int(scaled_value[1] * 255.999)
                self.fb[use_y, x, 2] = int(scaled_value[2] * 255.999)
                if self.depth_num == 4:
                    self.fb[use_y, x, 3] = int(scaled_value[3] * 255.999)
            else:
                self.fb[use_y, x, 0] = scaled_value[0]
                self.fb[use_y, x, 1] = scaled_value[1]
                self.fb[use_y, x, 2] = scaled_value[2]
                if self.depth_num == 4:
                    self.fb[use_y, x, 3] = scaled_value[3]


    def get_pixel(self, x, y):
        if self.origin == ORIGIN_LL:
            use_y = self.y_size - y - 1
        else:
            use_y = y

        if self.depth_num == 1:
            return self.fb[use_y, x]
        else: # self.depth_num == 3 or 4
            return self.fb[use_y, x, :]


def show_image(img: Image) -> None:
    # show the PIL image
    img.show()


def save_image(img: Image, filename: str) -> None:
    """
    Save the PIL image to disk.

    :param img: PIL Image data
    :param filename: filename to save
    """
    img.save(filename)


if __name__ == "__main__":
    # test framebuffer utilities
    fb = FrameBuffer(700, 512, np.int8, origin="UL")
    print(f'fb.shape={fb.shape}, fb.dtype={fb.dtype}')
    print(f'fb size=({fb.x_size}, {fb.y_size}), fb.dtype={fb.dtype}')

    # write to framebuffer
    for y in range(50,100):
        for x in range(40,120):
            fb.set_pixel(x, y, 128)  # or fb.fb[x, y] = 128

    # show framebuffer
    print(f'Pixel value at (10,10) = {fb.get_pixel(10, 10)}')  # or fb.fb[10,10]
    print(f'Pixel value at (70,70) = {fb.get_pixel(70, 70)}')
    img = fb.make_image()
    show_image(img)
    save_image(img, "framebuffer_test2.png")
    save_image(img, "framebuffer_test2.gif")
