import os
import tkinter as tk
from tkinter import font
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


class HomePage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # head
        label_head = tk.Label(self, text='ASL Recognition', font=("Helvetica", 25, "bold"))
        label_head.pack(padx=10, pady=10)

        # description
        description = "ASL Recognition using Webcam by capturing frames at certain interval using opencv and using deep learning to predict the word"
        lable_description = tk.Label(self, text=description, wraplength=500, justify="left",font=('Comic Sans MS', 12, 'bold italic'))
        lable_description.pack(padx=10, pady=10)

        # Model Selection
        model_option = "Select model for ASL recognition:"
        label_model_option = tk.Label(self, text=model_option, font=('Arial', 14, 'bold'))
        label_model_option.pack(padx=10, pady=10)

        option_variable_model = tk.StringVar(self)
        option_list_model = ['CNN Model', 'Vision Transformer']
        option_variable_model.set(option_list_model[0])

        option_menu_model = tk.OptionMenu(self, option_variable_model, *option_list_model)
        option_menu_model.pack(padx=10, pady=10)

        # Interval Selection
        interval_option = "Select interval for frame capture:"
        label_interval_option = tk.Label(self, text=interval_option, font=('Arial', 14, 'bold'))
        label_interval_option.pack(padx=10, pady=10)

        option_variable_interval = tk.StringVar(self)
        option_list_interval = [1, 2, 3 , 4, 5]
        option_variable_interval.set(option_list_interval[0])

        option_menu_interval = tk.OptionMenu(self, option_variable_interval, *option_list_interval)
        option_menu_interval.pack(padx=10, pady=10)

        button_start = tk.Button(self, text='Start', justify="center",
                                 command=lambda: controller.show_frame(CapturePage), font=('Arial', 12, 'bold'))
        button_start.pack(side=tk.BOTTOM, anchor="e", padx=10, pady=10)



class CapturePage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.cam = None
        self.img_counter = None
        label = tk.Label(self, text='ASL Recognition - Live Translation', font=('Comic Sans MS', 12, 'bold italic'))
        label.pack(pady=10, padx=10)

        self.image_x, self.image_y = 64, 64

        button_home = tk.Button(self, text='Home', command = lambda: controller.show_frame(HomePage), font=('Comic Sans MS', 12, 'bold italic'))
        button_home.pack(side="bottom", anchor="e", padx=10, pady=10)

        start_vid = tk.Button(self, height=2, width=20, text="Start Video", command=lambda: self.start_video())
        start_vid.pack()

        self.imageFrame = tk.Frame(self, width=600, height=500)
        self.imageFrame.pack()

        self.video_frame = tk.Label(self.imageFrame)
        self.video_frame.pack()

    def start_video(self):
        
        self.cam = cv2.VideoCapture(-1)
        print(self.cam.isOpened())

        self.img_counter = 0

        self.video_stream()
        self.video_frame.pack()

    def video_stream(self):

        self.img_counter += 1
        ret, frame = self.cam.read()
        frame = cv2.flip(frame, 1)
        print(frame)
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)
        # lower_blue = np.array([35, 10, 0])
        # upper_blue = np.array([160, 230, 255])
        # imcrop = img[102:298, 427:623]
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # cv2.putText(frame, "img_text", (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
        img_name = "tmp1.png"
        # save_img = cv2.resize(mask, (self.image_x, self.image_y))
        # cv2.imwrite(img_name, img)

        img = PIL.Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)
        self.video_frame.after(10, self.video_stream())






if __name__ == "__main__":

    app = MainApp()
    app.mainloop()