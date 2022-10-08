# -*- coding: utf-8 -*-

# import the necessary packages
from tkinter import Tk, Button, Label, filedialog
import base64
from tkinter import *
import grpc
from PIL import Image
from PIL import ImageTk
import numpy as np

import backend_pb2
import backend_pb2_grpc
#my package
import tkinter.messagebox

from tkinter import messagebox

#--------------------------------------

str_path = None
panelA = None
panelB = None

# initialize the window toolkit along with the two image panels
root = Tk()

#def run_model():
#    print("xd")
#    model = tf.keras.models.load_model('WilhemNet_86.h5')
#    print(model)
#    path_Model = backend_pb2.model(modelcnn=model)
#    modelPredict = backend_client.predict(path_Model)
#    tkinter.messagebox.showinfo( "Hello Python", modelPredict)

def select_image():
    # grab a reference to the image panels
    global panelA, backend_client, img_content, img_w, img_h, str_path
    # open a file chooser dialog and allow the user to select an input
    
    # image
    path = filedialog.askopenfilename()
    str_path = path
    # ensure a file path was selected
    if len(path) > 0:
        
        path_message = backend_pb2.img_path(path=path)
        response = backend_client.load_image(path_message)

        img_content = response.img_content
        img_w = response.width
        img_h = response.height

        b64decoded = base64.b64decode(img_content)
        image = np.frombuffer(b64decoded, dtype=np.uint8).reshape(img_h, img_w, -1)

        # convert the images to PIL format...
        image = Image.fromarray(image)
        
        
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)

        # if the panels are None, initialize them
        if panelA is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
            #update button Predict
            button1['state'] = 'normal'
        else:
            # update the pannels
            panelA.configure(image=image)
            panelA.image = image

def prediction():
    global panelB, backend_client, str_path

    #try:

    #image_data = backend_pb2.image_data(b64image=img_content, width=img_w, height=img_h)
    #data = backend_client.predict_data(image_data)

    if len(str_path) > 0:
        
        
        path_msg = backend_pb2.image_data(path2=str_path)
        response = backend_client.predict_data(path_msg)

        v_percent = response.proba
        v_result = response.label
        result_prediction = "Resultado de la evaluación de la imagen\n Presenta un tipo de neumonia {}, \n con una probabilidad de {:.2f}%".format(v_result, v_percent)

        messagebox.showinfo(title=None, message=result_prediction)


    else:
        messagebox.showerror('El path viene vacio')


root.title("Herramienta Detector Neumonía")
root.resizable(0, 0)


# Backend client definition
#Correction

maxMsgLength = 1024 * 1024 * 1024
options = [('grpc.max_message_length', maxMsgLength), ('grpc.max_send_message_length', maxMsgLength), ('grpc.max_receive_message_length', maxMsgLength)]
channel = grpc.insecure_channel("backend:50051", options = options)
backend_client = backend_pb2_grpc.BackendStub(channel=channel)
## Button predict
button1 = Button(root, text="Predecir", state='disabled', command=prediction) 
button1.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add th
# button the GUI
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
#text3 = Text(root)
#text3.place(BOTTOM)
# kick off the GUI
root.mainloop()
