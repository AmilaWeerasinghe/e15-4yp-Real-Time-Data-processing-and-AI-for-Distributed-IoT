import pickle
import socket
import json
import time
import numpy as np
import struct
import yolo as y
import weights as w

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
locking = 0
print('Socket Created')

s.bind(('localhost', 9999))

s.listen(1)  # number of backlog queues nothing to do with number od connections
# CHANGED # return the size of the struct,format L
payload_size = struct.calcsize("L")
data = b''  # binary output starts

# print(type(data)) #data turned to bytes


def send(c, data):
    data_string = pickle.dumps(data)
    message_size = struct.pack("L", len(data_string))
    c.sendall(message_size+data_string)


def receive_array(data, payload_size, conn):
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


while True:
    c, addr = s.accept()  # retunrs the socket connection object and address of client
    # receive data from client
    tic = time.process_time()
    data_variable = receive_array(data, payload_size, c)
    print('Connect with', addr, data_variable["data"].shape)
    print('position variable value recieved on server:', data_variable["pos"])
    print('position variable type:', type(data_variable["pos"]))
    # to lock sever computation to happen only once
    if(locking == 0):
        imgout = y.conv_forward(data_variable["data"], w.W1[:, :, :, (2 * data_variable["pos"]):],
                                w.b1[:, :, :, (2 * data_variable["pos"]):], data_variable["hpara"])
        out = {"data": imgout}
        send(c, out)
        locking = 1

    toc = time.process_time()
    print("Computation time for conv part2 = " + str(1000*(toc - tic)) + "ms")
    # send data to client
    #c.send(bytes("Welcome to server",'utf-8'))
    c.close()
