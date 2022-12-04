import os
import tkinter as tk
from tkinter import font, messagebox
from tkinter import ttk
import cv2
from PIL import ImageTk
import PIL.Image

import numpy as np


class MainApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.guidance_scale_value = None
        self.geometry("720x900")
        self.title("ASL Recognition")

        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for page in (HomePage, CapturePage):
            frame = page(self.container, self)
            self.frames[page] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(HomePage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def close_window(self):
        if messagebox.askokcancel("Quit", "would you like to quit"):
            self.destroy()

    def get_page(self, page_class):
        return self.frames[page_class]


class HomePage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        # head
        label_head = tk.Label(self, text='ASL Recognition', font=("Helvetica", 25, "bold"))
        label_head.pack(padx=10, pady=10)

        # description
        description = "ASL Recognition using Webcam by capturing frames at certain interval using opencv and using " \
                      "deep learning to predict the word "
        lable_description = tk.Label(self, text=description, wraplength=500, justify="left",
                                     font=('Comic Sans MS', 12, 'bold italic'))
        lable_description.pack(padx=10, pady=10)

        # Model Selection
        model_option = "Select model for ASL recognition:"
        label_model_option = tk.Label(self, text=model_option, font=('Arial', 14, 'bold'))
        label_model_option.pack(anchor="nw", padx=10, pady=10)

        self.variable_model = tk.StringVar(self)
        list_model = ['CNN Model', 'Vision Transformer']
        self.variable_model.set(list_model[0])

        for i, method in enumerate(list_model):
            model_button = tk.Radiobutton(self, text=method, variable=self.variable_model, value=method)
            model_button.pack(anchor='nw', padx=20, pady=10)

        # Interval Selection
        interval_option = "Select interval for frame capture:"
        label_interval_option = tk.Label(self, text=interval_option, font=('Arial', 14, 'bold'))
        label_interval_option.pack(anchor="nw", padx=10, pady=10)

        self.variable_interval = tk.IntVar(self)
        list_interval = [i for i in range(1, 6)]
        self.variable_interval.set(list_interval[0])

        for i, method in enumerate(list_interval):
            interval_button = tk.Radiobutton(self, text=method, variable=self.variable_interval, value=method)
            interval_button.pack(anchor='nw', padx=20, pady=10)

        # Buttons
        button_start = tk.Button(self, text='Start', justify="center", width=20,
                                 command=lambda: self.start_check(), font=('Arial', 14, 'bold'))
        button_start.pack(side=tk.LEFT, anchor="s", padx=10, pady=10)

        button_close = tk.Button(self, text='Close', width=20, justify="center",
                                 font=('Arial', 14, 'bold'), command=lambda: controller.close_window())
        button_close.pack(side=tk.RIGHT, anchor="s", padx=10, pady=10)

    def start_check(self):
        if self.variable_model.get() != "" and self.variable_interval.get() != 0:
            self.controller.show_frame(CapturePage)
        else:
            messagebox.showerror("Info", "Model or Interval not selected")


class CapturePage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.model = None
        self.frame_interval = None

        self.controller = controller
        self.HomePage = self.controller.get_page(HomePage)

        self.cam = None
        self.cam_on = False
        label = tk.Label(self, text='ASL Recognition - Live Translation', font=('Comic Sans MS', 12, 'bold italic'))
        label.pack(pady=10, padx=10)

        self.image_x, self.image_y = 64, 64

        self.start_vid = tk.Button(self, height=2, width=20, text="Start Video", command=lambda: self.start_video())
        self.start_vid.pack()

        self.imageFrame = tk.Frame(self, width=600, height=700)
        self.imageFrame.pack()

        self.video_frame = tk.Label(self.imageFrame)
        self.video_frame.pack()

        self.selected_frame = tk.Label(self.imageFrame)
        self.selected_frame.pack()

        self.stop_vid = tk.Button(self, height=2, width=20, text="Stop Video", command=lambda: self.stop_video())

        button_home = tk.Button(self, text='Home', width=20, justify="center", font=('Arial', 14, 'bold'),
                                command=lambda: [controller.show_frame(HomePage)])
        button_home.pack(side="left", anchor="s", padx=10, pady=10)

        button_close = tk.Button(self, text='Close', width=20, justify="center",
                                 font=('Arial', 14, 'bold'), command=lambda: controller.close_window())
        button_close.pack(side="right", anchor="s", padx=10, pady=10)

    def start_video(self):

        self.frame_interval = self.HomePage.variable_interval.get()
        self.cam = cv2.VideoCapture(-1)

        if self.cam.isOpened():
            self.cam_on = True
            self.stop_vid.pack(side="top", anchor="s", padx=10, pady=10)
            self.start_vid.forget()

        print(self.frame_interval)
        self.video_stream()

    def stop_video(self):
        if self.cam.isOpened():
            self.cam_on = False
            self.cam.release()
            self.start_vid.pack(side="top", anchor="s", padx=10, pady=10)
            self.stop_vid.forget()

    def video_stream(self):
        ret, frame = self.cam.read()
        frame = cv2.flip(frame, 1)
        # print(frame)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = cv2.rectangle(img, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)
        # lower_blue = np.array([35, 10, 0])
        # upper_blue = np.array([160, 230, 255])
        imcrop = img[102:298, 427:623]
        # hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # cv2.putText(frame, "img_text", (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
        # img_name = "tmp1.png"
        # save_img = cv2.resize(mask, (self.image_x, self.image_y))

        #
        cv2.imwrite("img.jpg", img)
        img = PIL.Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)

        cv2.imwrite("img_selected.jpg", imcrop)
        imcrop = PIL.Image.fromarray(imcrop)
        imgtk_selected_frame = ImageTk.PhotoImage(image=imcrop)
        self.selected_frame.imgtk = imgtk_selected_frame
        self.selected_frame.configure(image=imgtk_selected_frame)

        self.after(100, self.video_stream)


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
