import pickle
import json
import socket
import numpy as np
import struct
import time
from threading import *
import yolo as y
import weights as w


class client(Thread):
    payload_size = struct.calcsize("L")  # CHANGED #size of floats
    data = b''  # start with b for indicate bytes
    valdict = 0
    x = 0

    def __init__(self, values):
        super(client, self).__init__()
        self.valdict = values

    def send(self, c, data):
        data_string = pickle.dumps(data)
        message_size = struct.pack("L", len(data_string))
        c.sendall(message_size+data_string)

    def receive_array(self, data, payload_size, conn):
        while len(data) < payload_size:
            data += conn.recv(4096)

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]

    # Retrieve all data based on message size
        while len(data) < msg_size:
            data += conn.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

    # Extract frame
        frame = pickle.loads(frame_data)
        return frame

    def run(self):
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c.connect(('localhost', 9999))  # ip address and port
        self.send(c, self.valdict)
        self.x = self.receive_array(self.data, self.payload_size, c)

    def value(self):
        return self.x


# c.connect(('localhost',9999)) #ip address and port
# send data to server
# send(c,conv_dict)
# receive data from server
# out=receive_array(data,payload_size,c)
# print(out["data"].shape)
np.random.seed(1)  # always use the same initialized random numbers
# h256X256 image# here is the image input to detect
image = np.random.randn(1, 256, 256, 3)
# divide the weights shape to 3 so can take size to divide
a = round(w.W1.shape[3]/3)
conv_dict = {"data": image, "hpara": w.hparameters1, "pos": a}
#tic = time.process_time()
c = client(conv_dict)
c.start()
# client process conv portion
tic = time.process_time()
# divide the num of channels among nodes
out = y.conv_forward(image, w.W1[:, :, :, :a],
                     w.b1[:, :, :, :a], w.hparameters1)
toc = time.process_time()
print("Computation time conv part1 = " + str(1000*(toc - tic)) + "ms")
tic = time.process_time()
c.join()
print(c.value()["data"].shape)
toc = time.process_time()
out1 = np.concatenate((out, c.value()["data"]), axis=3)
#toc = time.process_time()
print("Computation time for join = " + str(1000*(toc - tic)) + "ms")
print("Out1 shape", out1.shape)
