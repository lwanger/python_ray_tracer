"""
GUI on top of the ray tracer

TODO:
    - add ambient light and background color
    - setting u,v on checkerboard squares?
    - set position of image on sphere (where u,v starts so can orient picture)
    - double-sided polys and back-side materials?
    - sizing on checkboard (sizing -- lower is bigger squares).
TODO:
    - update canvas paste a chunk
    - asyncio?
    - add settings dialog (image_name, size, aspect ratio, chunk size, # of worked, samples_per_pixel, max depth)
    - add multi-processing
    - why canvas bigger than rendering in X?

can use the following environment variables (or stored in .env file):

    USE_RES: low', 'med', 'high', or 'ultra'. Sets to the settings in res_settings. Each can be overwritten by the
        variables below.
    X_SIZE: x size of the rendered image
    ASPECT_RATIO: aspect ratio of the rendered image -- used to calculate y size (default is 16:9)
    SAMPLES_PER_PIXEL: samples per pixel
    SAMPLES_PER_LIGHT: samples per light
    MAX_DEPTH: maximum depth of bounces per pixel
    CHUNK_SIZE: size of chunks to calculate (e.g. value of 10 is 10x10 pixel blocks)
    RANDOM_CHUNKS: whether rendered chunks are in order or random (True - default)
    IMAGE_FILENAME: the file name to use to save the image

Len Wanger, copyright 2020
"""

from multiprocessing import Process, Pipe

from datetime import datetime
import numpy as np
from random import shuffle
import tkinter as tk

import dotenv
from PIL import Image, ImageTk

from rt import render_chunk, get_render_settings

from framebuffer import FrameBuffer, save_image, show_image
from geometry_classes import Vec3

from create_scene_funcs import *


# CREATOR_FUNC = create_simple_world
# CREATOR_FUNC = create_random_world
# CREATOR_FUNC = create_simple_world_2
# CREATOR_FUNC = create_simple_world_3
# CREATOR_FUNC = create_random_world2
# CREATOR_FUNC = create_checkerboard_world
# CREATOR_FUNC = create_checkerboard_world_2
# CREATOR_FUNC = create_image_texture_world
# CREATOR_FUNC = create_canonical_1  # ball over plane
# CREATOR_FUNC = create_canonical_2  # teapot
CREATOR_FUNC = create_stl_mesh
# CREATOR_FUNC = create_quad_world
# CREATOR_FUNC = create_disc_test_world


# messages from GUI
CANCEL_MSG = 100
SEND_PROGRESS_MSG = 101

# messages to GUI
PROGRESS_MSG = 200
COMPLETED_MSG = 201
CANCELLED_MSG = 202
CHUNK_RESULT_MSG = 203


class RenderCanceledException(BaseException):
    # Exception raised when the calculation process is cancelled
    pass


def receive_gui_message(pipe_conn):
    # receive a message from the GUI
    if pipe_conn and pipe_conn.poll():
        return pipe_conn.recv()

def send_gui_message(pipe_conn, event_id, data):
    # send a message to the GUI
    if pipe_conn:
        pipe_conn.send((event_id, data))


def render_worker(start, end, pipe_conn):
    # primes_in_range = []

    try:
        # for i in range(start, end):
        #     while True:
        #         response = receive_gui_message(pipe_conn)
        #
        #         if response is None:
        #             break
        #         elif response == CANCEL_MSG:
        #             raise CALC_CANCELLED
        #         elif response == SEND_PROGRESS_MSG:
        #             send_gui_message(pipe_conn, PROGRESS_MSG, (i - start) / (end - start))
        #
        #     if is_prime(i):
        #         primes_in_range.append(i)
        #
        # send_gui_message(pipe_conn, COMPLETED_MSG, primes_in_range)
        pass
    except RenderCanceledException:
        # send_gui_message(pipe_conn, CANCELLED_MSG, None)
        pass


class App(tk.Frame):
    def __init__(self):
        settings = get_render_settings()
        self.render_settings = settings
        self.x_size = settings['x_size']
        self.aspect_ratio = settings['aspect_ratio']
        self.chunk_size = settings['chunk_size']
        self.samples_per_pixel = settings['samples_per_pixel']
        self.max_depth = settings['max_depth']
        self.image_filename = settings['image_filename']
        self.random_chunks = settings['random_chunks']

        # self.gui_pipe_conn, self.worker_pipe_conn = Pipe()
        # self.worker = None

        self.y_size = settings['y_size']
        self.origin = settings['origin']
        self.horizontal = settings['horizontal']
        self.vertical = settings['vertical']
        self.lower_left = settings['lower_left']

        self.world_creator = CREATOR_FUNC(settings)

        self.create_gui()
        self.start_button_start = True  # False, means it's changed to cancel button
        self.image_saved = False
        self.fb = None
        self.render_cancelled = False


    def create_gui(self):
        self.root = tk.Tk()

        tk.Frame.__init__(self, self.root)
        self.root.wm_title("Ray Tracer")
        self.master.protocol("WM_DELETE_WINDOW", self.quit_cmd)

        # self.start_val = tk.IntVar()
        # self.start_val.set(1)

        self.status_str = tk.StringVar()
        self.status_str.set('')

        self.status_str2 = tk.StringVar()
        self.status_str2.set('')

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

        self.canvas = tk.Canvas(self.root, bg="#000000", width=self.x_size, height=self.y_size)
        self.canvas.grid(row=0, column=0, columnspan=3, sticky="nsew")

        self.start_button = tk.Button(self.root, text="Start", command=self.start_cmd)
        self.start_button.grid(row=1, column=2, sticky="nse", padx=5, pady=5)

        self.quit_button = tk.Button(self.root, text="Quit", command=self.quit_cmd)
        self.quit_button.grid(row=2, column=2, sticky="nse", padx=5, pady=5)

        self.status_label = tk.Label(self.root, textvariable=self.status_str, width=50)
        self.status_label.grid(row=1, column=0, sticky="new", columnspan=2, padx=5)

        self.status_label2 = tk.Label(self.root, textvariable=self.status_str2, width=50)
        self.status_label2.grid(row=2, column=0, sticky="new", columnspan=2, padx=3)


    def create_frame_buffer(self):
        # print(f'create_frame_buffer called')
        self.fb = FrameBuffer(self.x_size, self.y_size, np.int8, 'rgb')
        # self.pil_image = Image.new(mode="RGB", size=(X_SIZE, Y_SIZE), color=(128,128,128))
        # self.image = ImageTk.PhotoImage(self.pil_image)
        # self.canvas.create_image(X_SIZE, Y_SIZE, image=self.image)


    def run_gui(self):
        self.root.mainloop()


    def quit_cmd(self):
        # if self.worker:
        #     self.send_worker_message(CANCEL_MSG)
        #     self.worker.join(60)

        if self.fb is not None and self.image_saved is False:
            # print(f'Prompt to save image...')
            self.save_image()

        self.root.destroy()


    def save_image(self):
        img = self.fb.make_image()
        # show_image(img)
        save_image(img, self.image_filename)
        self.image_saved = True


    def send_worker_message(self, msg):
        """ Send a message to the worker process (send a msg on the worker input end of the pipe) """
        self.gui_pipe_conn.send(msg)


    def receive_worker_message(self):
        """ Receive a message from the worker process (receive a msg on the worker output end of the pipe)
            :return: message from the worker process
        """
        if self.gui_pipe_conn.poll():
            return self.gui_pipe_conn.recv()
        else:
            return None


    def process_worker_msgs(self):
        # Check every 100 ms if thread is done and process any messages in the queue.
        # print(f'process_worker_msgs called...')

        while True:
            break
            response = self.receive_worker_message()

            if response is None:
                break
            elif response[0] == PROGRESS_MSG:
                pct =int(response[1] * 100)
                self.status_str2.set(f'{pct}% done')
            elif response[0] == CHUNK_RESULT_MSG:
                TODO
            elif response[0] in {COMPLETED_MSG, CANCELLED_MSG}:
                self.start_button['state'] = "normal"
                self.cancel_button['state'] = "disabled"

                if response[0] == COMPLETED_MSG:
                    self.status_str.set(f'calculation complete')
                    self.status_str2.set(f'{len(response[1])} found')
                else:  # response == CANCELLED:
                    self.status_str.set(f'calculation cancelled')
                    self.status_str2.set('')

                self.worker.join()
                self.worker = None
                return

        self.root.after(100, self.process_worker_msgs)


    def update_worker_progress(self):
        # print(f'update_worker_progress called...')
        # Check every 500 ms ask for update of worker progress
        self.send_worker_message(SEND_PROGRESS_MSG)
        self.root.after(500, self.update_worker_progress)

    def start_render(self):
        # print(f'start_render called...')
        self.status_str.set(f'start_render called')
        self.root.update_idletasks()

        self.status_str.set(f'creating world...')
        self.root.update_idletasks()

        world = self.world_creator
        self.world = world['scene']
        self.camera = world['camera']

        self.create_frame_buffer()
        self.render_cancelled = False
        self.status_str.set(f'render started')
        self.status_str2.set('')
        self.start_button['text'] = "Cancel"
        self.start_button_start = False
        self.quit_button['state'] = "disabled"
        self.root.update_idletasks()
        self.render()
        self.quit_button['state'] = "normal"


    def finish_render(self, elapsed_time):
        # print(f'finish_render called...')
        ts = elapsed_time.total_seconds()

        if ts < 60:
            time_str = f'rendering time: {ts:.2f} seconds'
        elif ts < 3600:  # < 1 hr
            m, s = divmod(ts, 60)
            time_str = f'rendering time: {int(m)} minutes, {s:.2f} seconds'
        else:  # hours
            h, m = divmod(ts, 3600)
            m, s = divmod(m, 60)
            time_str = f'rendering time: {int(h)} hours, {int(m)} minutes, {s:.2f} seconds'

        im = self.fb.make_image()
        save_image(im, "rt_gui.png")
        self.status_str.set(f'render completed')
        self.status_str2.set(f'{time_str}')
        self.start_button['text'] = "Start"
        self.start_button_start = True
        self.quit_button['state'] = "normal"

        # self.pil_image = Image.new(mode="RGB", size=(X_SIZE, Y_SIZE), color=(128,128,128))
        # image = ImageTk.PhotoImage(im)
        # self.canvas.create_image(self.x_size, self.y_size, image=image)


    def cancel_render(self):
        # print(f'cancel_render called...')
        self.status_str.set(f'render cancelled')
        self.render_cancelled = True
        self.status_str2.set('')
        self.start_button['text'] = "Start"
        self.start_button_start = True
        self.quit_button['state'] = "normal"


    def start_cmd(self):
        # print(f'start_cmd called...')
        self.render_cancelled = False
        if self.start_button_start is True:  # start
            self.start_render()
        else:  # cancel
            self.cancel_render()

        # self.update_canvas(0, 0, None, None)

        # self.root.update_idletasks()  # required to get the label to update

        # args = (start, end, self.worker_pipe_conn)
        # self.worker = Process(target=calc_range_of_primes, args=args)
        # self.worker.start()
        # self.process_worker_msgs()
        # self.update_worker_progress()

    # def rgb_to_canvas_color(self, color):
    #     return f'{color[0]:02x}{color[1]:02x}{color[2]:02x}'

    def update_canvas(self, l: int, b: int, chunk_num: int, total_chunks: int):
        # print(f'update_canvas -- l={l}, b={b}, cn={chunk_num}, tc={total_chunks}')
        shape = self.fb.fb.shape
        # im = Image.frombytes("RGB", (shape[1],shape[0]), self.fb.fb.astype('b').tostring())
        self.im = Image.frombytes("RGB", (shape[1],shape[0]), self.fb.fb.astype('b').tostring())
        # photo = ImageTk.PhotoImage(image=im)
        self.photo = ImageTk.PhotoImage(image=self.im)
        # self.canvas.create_image(0,0,image=photo,anchor=tk.NW)
        self.canvas.create_image(0,0,image=self.photo,anchor=tk.NW)
        # self.status_str2.set(f'rendered scanline {y}')
        if chunk_num is not None:
            self.status_str2.set(f'rendered chunk {chunk_num} / {total_chunks}')

        self.canvas.update()
        # self.root.update()
        self.root.update_idletasks()


    def render(self):
        start_time = datetime.now()
        x_chunks, r = divmod(self.x_size, self.chunk_size)
        if r != 0:
            x_chunks += 1

        y_chunks, r = divmod(self.y_size, self.chunk_size)
        if r != 0:
            y_chunks += 1

        total_chunks = x_chunks * y_chunks
        chunk_num = 1

        try:
            if self.random_chunks is True:
                chunk_list = [ (i*self.chunk_size, j*self.chunk_size) for j in range(y_chunks) for i in range(x_chunks)]
                shuffle(chunk_list)

                for l,b in chunk_list:
                    r = l + self.chunk_size
                    t = b + self.chunk_size
                    render_chunk(self.world, self.camera, self.fb, self.x_size, self.y_size,
                                 l, b, r, t, self.samples_per_pixel, self.max_depth)
                    self.update_canvas(l, b, chunk_num, total_chunks)
                    chunk_num += 1
            else:
                for j in range(y_chunks):
                    for i in range(x_chunks):
                        l = i*self.chunk_size
                        r = l + self.chunk_size
                        b = j*self.chunk_size
                        t = b + self.chunk_size
                        render_chunk(self.world, self.camera, self.fb, self.x_size, self.y_size,
                                     l, b, r, t, self.samples_per_pixel, self.max_depth)
                        self.update_canvas(l, b, chunk_num, total_chunks)
                        chunk_num += 1

            end_time = datetime.now()
            elapsed_time = end_time - start_time
            self.finish_render(elapsed_time)
        except RenderCanceledException:
            self.cancel_render()


if __name__ == '__main__':
    dotenv.load_dotenv()
    app = App()
    app.run_gui()
