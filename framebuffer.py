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
    def __init__(self, x_size: int, y_size: int, dtype=np.int8, depth='s', origin="ll"):
        """
        Create a numpy array to act as a framebuffer.

        :param x_size:
        :param y_size:
        :param dtype: the data type for each pixel. Defaults to np.int8
        :param depth: the frame buffer depth - s - one value, rgb - 3 values per pixel, rgba - 4 vals per pixel
        :param origin: either "ll" (lower left - default) or "ul" upper left

        :return: numpy.Array
        """
        if origin == "ll":
            self.origin = ORIGIN_LL
        else:
            self.origin = ORIGIN_UL

        if depth == 's':
            self.fb = np.zeros(shape=(y_size, x_size), dtype=dtype)
            self.depth_num = 1
        elif depth == 'rgb':
            self.fb = np.zeros(shape=(y_size, x_size, 3), dtype=dtype)
            self.depth_num = 3
        elif depth == 'rgba':
            self.fb = np.zeros(shape=(y_size, x_size, 4), dtype=dtype)
            self.depth_num = 4
        else:
            raise RuntimeError(f'Invalid FrameBuffer depth ({depth})')

        self.x_size = x_size
        self.y_size = y_size
        self.dtype = dtype
        self.depth = depth


    def get_x_size(self):
        return self.x_size

    def get_y_size(self):
        return self.y_size

    def get_depth(self):
        return self.y_size

    def get_shape(self):
        return self.fb.shape

    def make_image(self, mode="L") -> Image:
        """
        Create a PIL (pillow) image from a numpy array/framebuffer.

        :param mode: PIL mode - "L" for luminance/grayscale, "RGB", "RGBA", "1" binary, "P" 8 bit palette
        :return: PIL.Image
        """
        if self.depth_num == 1:
            return Image.fromarray(self.fb, mode="L")
        elif self.depth_num == 3:
            return Image.fromarray(self.fb, mode="RGB")
        else:  # depth_num == 4
            return Image.fromarray(self.fb, mode="RGBA")

    def set_pixel(self, x, y, value, samples=1):
        if samples == 1:
            scaled_value = value
        else:
            scale = 1 / samples
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
