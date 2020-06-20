# CS_292

In 1980 Ed Catmull and a number of the computer graphics wizards from Lucasfilm (Jim Blinn, Loren Carpenter, and Alvy Ray Smith) taught a computer graphics class at Berkeley. THis class is on Youtube at:https://www.youtube.com/channel/UCNXre0qpHjdhC29xH8WkKnw.

I thought it would be fun to try to write code for the concepts there. This repository is my attempt.

Our modern tools are very luxurious compared to what they had available. Modern hardware and software tools like Python and NumPy mad this very easy in comparison to C on PDP 11's.

First is a Python/NumPy class to act as a framebuffer. The FrameBuffer class has some simple methods:

__init__(x, y, dtype) -- x size, y size and data type (see numpy for types). defaults to type uint8 for ints 0..256
get_x_size() -- returns the X size of the framebuffer
get_y_size() -- returns the Y size of the framebuffer
set_pixel(x, y, val) -- set the pixel at location x, y to the value (val)
get_pixel(x,y) -- return the value of the pixel
make_image(mode) -- creates an image using PIL. For list of modes see pillow documentation. Defaults to "L" - 0..256 luminosity value. There are two functions (show_image and save_image) that can take the return value and show it on screen or save it to disk.

# Also in here is ray tracing from Pete Shirley's _Ray Tracing in One Weekend_ book:

for example, run: python listing_55.py

optional environment variables:

IMAGE_FILE  (e.g. export IMAGE_FILE="image.png")
USE_RES (eg. export USE_RES=2) - 0 - minimal, 1 - low-res, 2, normal, 3- high-res

Lots of fun stuff to do on the ray tracing:

- profile / optimize
- multi-processing
- GUI - cancel button, set params, pre-view rendering
- plane geometry
- triangle geometry
- axis-aligned box (AABB) geometry
- spatial data structure for speed ups
- read in STL (or STEP)
- texture mapping
- lights
- shadows
- de-noising
- ...

Len Wanger -- 2020
