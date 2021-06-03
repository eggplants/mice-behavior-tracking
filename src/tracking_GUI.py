import time
# import tkinter as tk
from datetime import datetime

import cv2
import easygui
import numpy as np
import serial
from serial.tools import list_ports

BANNER = '''\
           _                _                  _    _
 _ __ ___ (_) ___ ___      | |_ _ __ __ _  ___| | _(_)_ __   __ _
| '_ ` _ \| |/ __/ _ \_____| __| '__/ _` |/ __| |/ / | '_ \ / _` |
| | | | | | | (_|  __/_____| |_| | | (_| | (__|   <| | | | | (_| |
|_| |_| |_|_|\___\___|      \__|_|  \__,_|\___|_|\_\_|_| |_|\__, |
                                                            |___/
'''


def show_window(title: str, message: str, exit_: bool = False) -> None:
    easygui.msgbox(title, message)
    if exit_:
        exit(1)


class DeviceInfo():
    def __init__(self):
        self.num = 0


class TrackingError(Exception):
    pass

# def get_device_num(message, device_info):
#     root = tk.Tk()
#     frame = tk.Frame(root)
#     label = tk.Label(frame, text=message)
#     t = tk.IntVar()
#     entry = tk.Entry(frame, textvariable=t)

#     def onclick():
#         num = t.get()
#         device_info.num = num
#         root.destroy()

#     button = tk.Button(frame, text='OK', command=onclick)
#     frame.pack()
#     label.pack(side=tk.LEFT)
#     entry.pack(side=tk.LEFT)
#     button.pack(side=tk.LEFT)
#     root.mainloop()


def get_ans(question, selections=["y", "n"]):
    reply = input(question)
    selections = list(map(str,  selections))
    if reply in selections:
        return reply
    else:
        return get_ans("invalid answer. retry: ", selections)


# def raise_error(message):
#     root = tk.Tk()
#     frame = tk.Frame(root)
#     label = tk.Label(frame, text=message)

#     def onclick():
#         root.destroy()
#         exit()

#     button = tk.Button(frame, text='OK', command=onclick)
#     frame.pack()
#     label.pack(side=tk.LEFT)
#     button.pack(side=tk.LEFT)
#     root.mainloop()


def strlist_to_comma_str(lis):
    return ", ".join(map(str, lis))


def select_port():
    ser = serial.Serial()
    ser.baudrate = 9600  # same as Serial.begin in Arduino
    ser.timeout = 0.1
    print('[Serial Device]')
    print('Checking serial devices... (baudrate: {}, timeout: {})'.format(
        ser.baudrate, ser.timeout))

    ports = list_ports.comports()  # get port data
    devices = [info.device for info in ports]
    if len(devices) == 0:
        show_window(str(type(TrackingError)),
                    "Error: serial device not found")
        raise TrackingError('Error: serial device not found')
    elif len(devices) == 1:
        print('=> Only found: %s' % devices[0])
        ser.port = devices[0]
    else:
        print('=> Some found:')
        for i in range(len(devices)):
            print('%3d: open %s' % (i, devices[i]))
        device_info = DeviceInfo()
        dev_comma_sep = strlist_to_comma_str([*range(len(devices))])
        device_num = get_ans('Select one target port ({}): '.format(
            dev_comma_sep), [*range(len(devices))])
        device_info.num = int(device_num)
        ser.port = devices[device_info.num]

    try:
        ser.open()
        return ser
    except Exception as e:
        show_window(str(type(e)), "Error: occurs when opening serial")
        raise TrackingError('Error: error occurs when opening serial')

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
        self.centerX = []
        self.centerY = []
        self.centlist = []

    def center_opelation(self, mb, i):
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mb)
        try:
            cx = int(centroids[1+np.nanargmax(stats[1:, -1])][0])
            cy = int(centroids[1+np.nanargmax(stats[1:, -1])][1])

            self.centerX.append(cx)
            self.centerY.append(cy)
            self.centlist.append(
                (self.centerX[-2] - self.centerX[-1])**2 +
                (self.centerY[-2] - self.centerY[-1])**2)
        except Exception:
            self.centlist.append(0)
        # cation cahnegd
        return 300*np.abs(
            (self.centlist[i] - np.mean(self.centlist))/np.std(self.centlist))


def get_cams_list():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    cv2.destroyAllWindows()
    return arr


def video_body(ser, C):
    print('[Camera Device]')
    # set cam device num:
    camera_info = DeviceInfo()
    print('Checking camera devices...')
    cam_list = get_cams_list()
    print('=>', cam_list)
    device_num = int(
        get_ans('Select one camera device ({}): '.format(
            strlist_to_comma_str(cam_list)),
            cam_list))
    camera_info.num = device_num

    print('[Options]')
    # set if save:
    if get_ans('Save video(.avi) ? (y/n): ') == 'y':
        save_video = True
    else:
        save_video = False

    if get_ans('Save result(.csv)? (y/n): ') == 'y':
        save_csv = True
    else:
        save_csv = False

    show_window('If you quit, type "q"', 'info', False)
    cap = cv2.VideoCapture(camera_info.num)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    nowtime = str(datetime.now())
    filename = nowtime.replace(' ', '').replace(
        '.', '-').replace(':', '-') + str('.avi')
    csvfilename = nowtime.replace(' ', '').replace(
        '.', '-').replace(':', '-') + str('.csv')
    out = ''
    if save_video:
        out = cv2.VideoWriter(filename, fourcc, 10.0, (1280, 480))

    kernel1 = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((10, 10), np.uint8)
    if save_csv:
        csvfile = open(csvfilename, 'w')
    else:
        csvfile = None

    t0 = time.perf_counter()
    i = 0
    while True:
        ret, frame = cap.read()
        if ret:
            i += 1
            t1 = time.perf_counter()
            # image processing
            two = np.where(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                           < 5, 254, 0).astype('uint8')
            cl = cv2.morphologyEx(two, cv2.MORPH_CLOSE, kernel1)
            op = cv2.morphologyEx(cl, cv2.MORPH_OPEN, kernel2)
            mb = cv2.medianBlur(op, 5)

            C.mb = mb
            info = C.center_opelation(mb, i)

            info_str = str(info)
            ser.write(info_str.encode('utf-8'))
            ser.write(b'\n')

            timestamp = str(datetime.now())[0:21].replace(' ', '')
            infos = info_str[0:6] + ',' + \
                str(int((t1-t0)/10)) + ',' + timestamp
            ser.reset_output_buffer()
            # allinfo = info_str + ',' + str(int((t1-t0)/10))  + ',' +timestamp
            print(infos)
            if save_csv:
                csvfile.write(infos)
                csvfile.write('\n')

            cv2.putText(frame, infos, (40, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), thickness=2)
            try:
                # put circle
                cv2.circle(mb, (C.centerX[i], C.centerY[i]),
                           10, (150, 150, 150),  thickness=4)
            except Exception:
                # last point
                cv2.circle(mb, (C.centerX[-1], C.centerY[-1]),
                           10, (150, 150, 150), thickness=4)

            out_frame_color = cv2.cvtColor(cv2.hconcat(
                [mb, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)]),
                cv2.COLOR_GRAY2BGR)
            if save_video and out != '':
                out.write(out_frame_color)
            cv2.imshow('frames', out_frame_color)

            t2 = time.perf_counter()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif t2 - t1 > 25200:
                break
        else:
            show_window(str(type(TrackingError)),
                        "Error: error occurs when get frame from camera")
            raise TrackingError(
                'Error: error occurs when get frame from camera')

    csvfile.close()
    cap.release()
    if out != '':
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print(BANNER)
    ser = select_port()
    # show_connection(ser)

    # init center class
    C = mouseinfo()
    C.centerX = [0, 0]
    C.centerY = [0, 0]
    C.centlist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    video_body(ser, C)
