from concurrent import futures
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

import grpc
import cv2

import backend_pb2
import backend_pb2_grpc


class BackendService(backend_pb2_grpc.BackendServicer):
    def _test_func(self, path):
        image = cv2.imread(path)
        h, w, _ = image.shape
        print(f"image shape: w-{w} h-{h}")

        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_str = base64.b64encode(image)
        return image_str, w, h

    def load_image(self, request, context):
        # load the image from disk
        path = request.path
        print(path)

        image_str, w, h = self._test_func(path=path)

        response_message = backend_pb2.image(img_content=image_str, width=w, height=h)
        return response_message

    def _read_img(self, path):
    
        img = cv2.imread(path)
        img_array = np.asarray(img)
        img2 = img_array.astype(float) 
        img2 = (np.maximum(img2,0) / img2.max()) * 255.0
        img2 = np.uint8(img2)
        
        return img2
        

    def predict_data(self, request, context):

        #before
        #b64decoded = base64.b64decode(request.b64image)
        #image = np.frombuffer(b64decoded, dtype=np.uint8).reshape(request.height, request.width, -1)
        path2 = request.path2
        
        arreglo_path2 = self._read_img(path=path2)
        #   1. call function to pre-process image: it returns image in batch format
        batch_array_img = self.preprocess(arreglo_path2)
        #   2. call function to load model and predict: it returns predicted class and probability
        model = self.model_fun()
        # model_cnn = tf.keras.models.load_model('conv_MLP_84.h5')
        prediction = np.argmax(model.predict(batch_array_img))
        proba = np.max(model.predict(batch_array_img)) * 100
        label = ""
        if prediction == 0:
            label = "bacteriana"
        if prediction == 1:
            label = "normal"
        if prediction == 2:
            label = "viral"
        response_message = backend_pb2.image_prediction(label=label, proba=proba)
        return response_message
        
    def preprocess(self, array):
        array = cv2.resize(array, (512, 512))
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        array = clahe.apply(array)
        array = array / 255
        array = np.expand_dims(array, axis=-1)
        array = np.expand_dims(array, axis=0)
        return array

    def model_fun(self):
        model_cnn = tf.keras.models.load_model('WilhemNet_86.h5')
        return model_cnn


def serve():
    maxMsgLength = 1024 * 1024 * 1024
    options = [('grpc.max_message_length', maxMsgLength),('grpc.max_send_message_length', maxMsgLength),('grpc.max_receive_message_length', maxMsgLength)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    backend_pb2_grpc.add_BackendServicer_to_server(BackendService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()