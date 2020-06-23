"""
GUI on top of the ray tracer

TODO:
    - update canvas paste a chunk
    - asyncio?
    - add import create_world
    - add settings dialog (image_name, size, aspect ratio, chunk size, # of worked, samples_per_pixel, max depth)
    - add multi-processing
    - why canvas bigger than rendering in X?

Len Wanger, copyright 2020
"""

from multiprocessing import Process, Pipe

from datetime import datetime
import numpy as np
from random import random, uniform, shuffle
import tkinter as tk
from PIL import Image, ImageTk

from framebuffer import FrameBuffer, save_image, show_image
from geometry_classes import Vec3, Ray

from weekend_final_pic import Camera, ray_color, create_random_world, create_simple_world

X_SIZE = 384
X_SIZE = 100
# CHUNK_SIZE = 100
CHUNK_SIZE = 10
# CHUNK_SIZE = 5
RANDOM_CHUNKS = True

# SAMPLES_PER_PIXEL = 100
# SAMPLES_PER_PIXEL = 50
SAMPLES_PER_PIXEL = 10
# MAX_DEPTH = 50
MAX_DEPTH = 25
FOV = 20

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

###


# def ray_color(ray: Ray):
#     unit_dir = ray.direction.unit_vector()
#     t = 0.5 * (unit_dir.y + 1.0)
#     return Vec3(1.0,1.0,1.0)*(1.0-t) + Vec3(0.5,0.7,1.0)*t


class App(tk.Frame):
    def __init__(self):
        # self.gui_pipe_conn, self.worker_pipe_conn = Pipe()
        # self.worker = None

        # self.world_creator = create_simple_world
        self.world_creator = create_random_world

        aspect_ratio = 16.0 / 9.0
        self.x_size = X_SIZE
        self.y_size = int(X_SIZE / aspect_ratio)

        self.samples_per_pixel = SAMPLES_PER_PIXEL
        self.max_depth = MAX_DEPTH

        viewport_height = 2.0
        viewport_width = aspect_ratio * viewport_height
        focal_length = 1.0
        self.chunk_size = CHUNK_SIZE

        self.look_from = Vec3(13, 2, 3)
        self.look_at = Vec3(0, 0, 0)
        self.vup = Vec3(0, 1, 0)
        self.fd = 10.0
        self.aperature = 0.1
        self.fov = 20

        self.origin = Vec3(0.0, 0.0, 0.0)
        self.horizontal = Vec3(viewport_width, 0, 0)
        self.vertical = Vec3(0, viewport_height, 0)
        self.lower_left = self.origin - self.horizontal.div_val(2) - self.vertical.div_val(2) - Vec3(0, 0, focal_length)

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
        self.fb = FrameBuffer(X_SIZE, self.y_size, np.int8, 'rgb')
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
        save_image(img, "rt_gui.png")
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

        self.camera = Camera(self.look_from, self.look_at, self.vup, self.fov, aperature=self.aperature, focus_dist=self.fd)
        self.status_str.set(f'creating world...')
        self.root.update_idletasks()

        self.world = self.world_creator()
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
            time_str = f'rendering time: {m} minutes, {s:.2f} seconds'
        else:  # hours
            h, m = divmod(ts, 3600)
            m, s = divmod(m, 60)
            time_str = f'rendering time: {h} hours, {m} minutes, {s:.2f} seconds'

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


    def render_chunk(self, l:int, b:int, r: int, t: int):
        use_r = min(r, self.x_size)
        use_t = min(t, self.y_size)
        # print(f'l={l}, r={r}, b={b}, t={t}, use_r={use_r}, use_t={use_t}')

        for j in range(b, use_t):
            for i in range(l, use_r):
                pixel_color = Vec3(0, 0, 0)

                for s in range(self.samples_per_pixel):
                    if self.render_cancelled is True:
                        raise RenderCanceledException

                    u = (i + random()) / (self.x_size - 1)
                    v = (j + random()) / (self.y_size - 1)
                    ray = self.camera.get_ray(u, v)
                    pixel_color += ray_color(ray, self.world, self.max_depth)

                self.fb.set_pixel(i, j, pixel_color.get_unscaled_color(), self.samples_per_pixel)


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
            if RANDOM_CHUNKS is True:
                chunk_list = [ (i*self.chunk_size, j*self.chunk_size) for j in range(y_chunks) for i in range(x_chunks)]
                shuffle(chunk_list)

                for l,b in chunk_list:
                    r = l + self.chunk_size
                    t = b + self.chunk_size
                    self.render_chunk(l, b, r, t)
                    self.update_canvas(l, b, chunk_num, total_chunks)
                    chunk_num += 1
            else:
                for j in range(y_chunks):
                    for i in range(x_chunks):
                        l = i*self.chunk_size
                        r = l + self.chunk_size
                        b = j*self.chunk_size
                        t = b + self.chunk_size
                        self.render_chunk(l, b, r, t)
                        self.update_canvas(l, b, chunk_num, total_chunks)
                        chunk_num += 1

            end_time = datetime.now()
            elapsed_time = end_time - start_time
            self.finish_render(elapsed_time)
        except RenderCanceledException:
            self.cancel_render()


if __name__ == '__main__':
    app = App()
    app.run_gui()


if __name__ == '__main__':
    pass