from datetime import datetime
import gc
import time

import cv2
import numpy as np
import serial
from serial.tools import list_ports 
import tkinter as tk

class DeviceInfo():
    def __init__(self):
        self.num = None

def get_device_num(message, device_info):
    root = tk.Tk()
    frame = tk.Frame(root)
    label = tk.Label(frame, text=message)
    t = tk.IntVar()
    entry = tk.Entry(frame, textvariable=t)

    def onclick():
        num = t.get()
        device_info.num = num
        root.destroy()

    button = tk.Button(frame, text='OK', command=onclick)
    frame.pack()
    label.pack(side=tk.LEFT)
    entry.pack(side=tk.LEFT)
    button.pack(side=tk.LEFT)
    root.mainloop()

def raise_error(message):
    root = tk.Tk()
    frame = tk.Frame(root)
    label = tk.Label(frame, text=message)

    def onclick():
        root.destroy()
        exit()

    button = tk.Button(frame, text='OK', command=onclick)
    frame.pack()
    label.pack(side=tk.LEFT)
    button.pack(side=tk.LEFT)
    root.mainloop()

def select_port():
    ser = serial.Serial()
    ser.baudrate = 9600 # same as Serial.begin in Arduino
    ser.timeout = 0.1
    print('chaking serial', ser)

    ports = list_ports.comports() # get port data
    devices = [info.device for info in ports]
    if len(devices) == 0:
        raise_error('error: serial device not found')
    elif len(devices) == 1:
        print('only found %s' % devices[0])
        ser.port = devices[0]
    else:
        for i in range(len(devices)):
            print('%3d: open %s' % (i,devices[i]))
        device_info = DeviceInfo()
        get_device_num('please input number of target port:', device_info)
        ser.port = devices[device_info.num]

    try:
        ser.open()
        return ser
    except:
        raise_error('error occurs when opening serial')

# check connection
def show_connection(ser):
    t = 0
    while t < 3:
        startshow = str(60000)
        ser.write(startshow.encode('utf-8'))
        ser.write(b'\n')
        ser.reset_output_buffer()
        time.sleep(0.5)
        
        startshow_2 = str(0)
        ser.write(startshow_2.encode('utf-8'))
        ser.write(b'\n')
        ser.reset_output_buffer()
        time.sleep(0.5)

        t += 1

class mouseinfo():
    def init(self):
        self.mb = []
        self.centerX   = []
        self.centerY   = []
        self.centlist  = []

    def center_opelation(self, mb, i):
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mb)
        try:
            cx = int(centroids[1+np.nanargmax(stats[1:, -1])][0])
            cy = int(centroids[1+np.nanargmax(stats[1:, -1])][1])

            self.centerX.append(cx)
            self.centerY.append(cy)
            self.centlist.append((self.centerX[-2] - self.centerX[-1])**2 + (self.centerY[-2] - self.centerY[-1])**2 ) 
        except:
            self.centlist.append(0)
        return 300*np.abs((self.centlist[i] - np.mean(self.centlist))/np.std(self.centlist)) # cation cahnegd

def video_body(ser, C):
    camera_info = DeviceInfo()
    get_device_num('please input number of target camera device:', camera_info)

    cap = cv2.VideoCapture(camera_info.num)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    nowtime = str(datetime.now())
    filename = nowtime.replace(' ', '').replace('.', '-').replace(':', '-') + str('.avi')
    csvfilename = nowtime.replace(' ', '').replace('.', '-').replace(':', '-') + str('.csv') 
    out = cv2.VideoWriter(filename, fourcc, 10.0, (1280, 480))

    kernel1 = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((10, 10), np.uint8)
    csvfile = open(csvfilename,'w') 

    t0 = time.perf_counter()
    i = 0
    while True:
        ret, frame = cap.read()
        if ret:
            i+=1
            t1 = time.perf_counter()
            # image processing
            two = np.where(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) < 5,254,0).astype('uint8')
            cl = cv2.morphologyEx(two,cv2.MORPH_CLOSE,kernel1) 
            op = cv2.morphologyEx(cl,cv2.MORPH_OPEN,kernel2)  
            mb = cv2.medianBlur(op,5)

            C.mb=mb
            info = C.center_opelation(mb,i)
            
            info_str = str(info) 
            ser.write(info_str.encode('utf-8'))
            ser.write(b'\n')

            timestamp = str(datetime.now())[0:21].replace(' ','')
            infos =  info_str[0:6] + ',' + str(int((t1-t0)/10)) + ',' + timestamp
            ser.reset_output_buffer()
            #allinfo = info_str + ',' + str(int((t1-t0)/10))  + ',' +timestamp
            print(infos)
            csvfile.write(infos)
            csvfile.write('\n')

            
            cv2.putText(frame,infos, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
            try:                
                cv2.circle(mb,(C.centerX[i],C.centerY[i]),10,(150,150,150),  thickness=4) # put circle
            except:
                cv2.circle(mb,(C.centerX[-1],C.centerY[-1]),10,(150,150,150),thickness=4) # last point

            out_frame_color = cv2.cvtColor(cv2.hconcat([mb,cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)]),cv2.COLOR_GRAY2BGR)            
            out.write(out_frame_color) 
            cv2.imshow('frames',out_frame_color)

            t2 = time.perf_counter()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif t2 - t1 > 25200:
                break
        else:
            raise_error('error occurs when get frame from camera')
    
    csvfile.close()
    cap.release() 
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    ser=select_port()
    # show_connection(ser)

    # init center class
    C=mouseinfo()
    C.centerX  = [0,0]
    C.centerY  = [0,0]
    C.centlist = [0,0,0,0,0,0,0,0,0,0,0]
    video_body(ser,C)
