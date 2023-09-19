import tensorflow
import tkinter as tk 
from tkinter import filedialog
import numpy as np 

from tensorflow import keras 
from keras.preprocessing import image
from keras.utils import load_img
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os

import cv2
from tkinter import *
from PIL import ImageTk, Image
import _tkinter

win = tk.Tk()

def b1_click():
    global path2
    try:
        json_file = open('model1.json','r')
        loaded_json_model = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_json_model)
        loaded_model.load_weights("model1.h5")
        print('Loaded model from disk')
        label=["blackspot","canker"]
        path2 = filedialog.askopenfilename()
        print(path2)
        
        test_image = tensorflow.keras.utils.load_img(path2,target_size=(128,128))
        test_image = tensorflow.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis=0)
        result = loaded_model.predict(test_image)
        fresult = np.max(result)
        label2 = label[fresult.argmax()]
        print(label2)
        lb1.configure(text=label2)
        win.mainloop()
    except IOError:
        pass

label1 = Label(win,text="GUI for Citrus Disease Detection",fg="blue")
label1.pack()

b1 = tk.Button(win,text='Browse the test image',width=25,height=3,fg='red',command=b1_click)
b1.pack()
lb1=Label(win,text="Result",fg='blue')
lb1.pack()

win.geometry("600x300")
win.title("Citrus Fruit Disease Detection Using CNN")
win.bind("<Return>",b1_click)
win.mainloop()