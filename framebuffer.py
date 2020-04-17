"""
Frame buffer utilities

Len Wanger -- 2020
"""

import numpy as np
from PIL import Image

class FrameBuffer():
    def __init__(self, x_size: int, y_size: int, dtype=np.int8):
    # def make_framebuffer(x_size: int, y_size: int, dtype=np.int8) -> np.array:
        """
        Create a numpy array to act as a framebuffer.

        :param x_size:
        :param y_size:
        :param dtype: the data type for each pixel. Defaults to np.int8

        :return: numpy.Array
        """
        self.fb = np.zeros(shape=(x_size, y_size), dtype=dtype)
        self.x_size = x_size
        self.y_size = y_size
        self.dtype = dtype

    def get_shape(self):
        return self.fb.shape

    shape = property(get_shape)

    def make_image(self, mode="L") -> Image:
        """
        Create a PIL (pillow) image from a numpy array/framebuffer.

        :param mode: PIL mode - "L" for luminance/grayscale, "RGB", "RGBA", "1" binary, "P" 8 bit palette
        :return: PIL.Image
        """
        return Image.fromarray(self.fb, mode="L")

    def set_pixel(self, x, y, value):
        self.fb[x,y] = value

    def get_pixel(self, x, y):
        return self.fb[x,y]


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
    fb = FrameBuffer(512, 512, np.int8)
    print(f'fb.shape={fb.shape}, fb.dtype={fb.dtype}')
    print(f'fb size=({fb.x_size}, {fb.y_size}), fb.dtype={fb.dtype}')

    # write to framebuffer
    for y in range(50,100):
        for x in range(50,100):
            fb.set_pixel(x, y, 128)  # or fb.fb[x, y] = 128

    # show framebuffer
    print(f'Pixel value at (10,10) = {fb.get_pixel(10, 10)}')  # or fb.fb[10,10]
    print(f'Pixel value at (70,70) = {fb.get_pixel(70, 70)}')
    img = fb.make_image()
    show_image(img)
    save_image(img, "framebuffer_test.png")
    save_image(img, "framebuffer_test.gif")
