import getpass
import time
import warnings
from datetime import datetime
from typing import IO, Any, Dict, List, NewType, Optional, Union

import cv2
import numpy as np
import serial
from serial.tools import list_ports

warnings.filterwarnings("ignore")
Serial = NewType(
    'Serial', Union[serial.serialwin32.Serial,
                    # serial.serialposix.Serial,
                    # serial.serialjava.Serial
                    ])

BANNER = '''\
           _                _                  _    _
 _ __ ___ (_) ___ ___      | |_ _ __ __ _  ___| | _(_)_ __   __ _
| '_ ` _ \\| |/ __/ _ \\_____| __| '__/ _` |/ __| |/ / | '_ \\ / _` |
| | | | | | | (_|  __/_____| |_| | | (_| | (__|   <| | | | | (_| |
|_| |_| |_|_|\\___\\___|      \\__|_|  \\__,_|\\___|_|\\_\\_|_| |_|\\__, |
                                                            |___/
v0.10
'''


class DeviceInfo():
    def __init__(self):
        self.num: int = 0


class TrackingError(Exception):
    pass


class MouseInfo():
    def init(self):
        self.med_blur = []
        self.centerX = []
        self.centerY = []
        self.centlist = []

    def center_operation(self, med_blur: cv2.medianBlur, idx: int):
        """Extract controids of connected components

        search conn-cmpt and index them
        returns: nlabels, labels, stats, centroids
        nlabels: max index of conn-cmpt
        labels: list of labeled pixels
        stats: list of conn-cmpts' box info
        centroids: list of center of gravity
        """
        _, _, stats, centroids = cv2.connectedComponentsWithStats(med_blur)
        try:
            cx = int(centroids[1+np.nanargmax(stats[1:, -1])][0])
            cy = int(centroids[1+np.nanargmax(stats[1:, -1])][1])

            self.centerX.append(cx)
            self.centerY.append(cy)
            self.centlist.append(
                (self.centerX[-2] - cx)**2 + (self.centerY[-2] - cy)**2)
        except ValueError:
            self.centlist.append(0)

        # returns: 300 * z-value
        # zvalue is alse called standard score
        z = (self.centlist[idx] - np.mean(self.centlist))/np.std(self.centlist)
        return 300 * np.abs(z)


def get_ans(question: str, selections: List[str] = ['y', 'n']):
    """Question and receive a selection"""
    reply = input(question)
    selections = list(map(str,  selections))
    if reply in selections:
        return reply
    else:
        return get_ans('invalid answer. retry: ', selections)


def get_cams_list() -> List[int]:
    """Get a list of camera devices"""
    idx = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        else:
            arr.append(idx)
        cap.release()
        idx += 1
    cv2.destroyAllWindows()
    return arr


def list_to_comma_str(lis: List[Any]) -> str:
    return ', '.join(map(str, lis))


def show_window(title: str, message: str,
                button: str = 'PRESS ENTER KEY', exit_: bool = False) -> None:
    """Show infomation on terminal"""
    print('[{}]\n{}'.format(title, message))
    getpass.getpass('[{}]'.format(button))
    # import easygui
    # easygui.msgbox(title, message)
    if exit_:
        exit(1)


# def show_connection(ser) -> None:
#     for _ in range(3):
#         startshow = str(60000)
#         ser.write(startshow.encode('utf-8'))
#         ser.write(b'\n')
#         ser.reset_output_buffer()
#         time.sleep(0.5)

#         startshow_2 = str(0)
#         ser.write(startshow_2.encode('utf-8'))
#         ser.write(b'\n')
#         ser.reset_output_buffer()
#         time.sleep(0.5)


def select_port() -> Serial:
    """Check port devices with serial communication and select"""
    ser: Serial = serial.Serial()
    ser.baudrate = 9600  # same as Serial.begin in Arduino
    ser.timeout = 0.1
    print('[Serial Device]')
    print('Checking serial devices... (baudrate: {}, timeout: {})'.format(
        ser.baudrate, ser.timeout))

    ports = list_ports.comports()  # get port data
    devices = [info.device for info in ports]
    if len(devices) == 0:
        show_window(str(type(TrackingError)),
                    'Error: serial device not found', exit_=True)
        raise TrackingError('Error: serial device not found')
    elif len(devices) == 1:
        print('=> Only found: %s' % devices[0])
        ser.port = devices[0]
    else:
        print('=> Some found:')
        print_devices(devices)
        device_info = DeviceInfo()
        dev_comma_sep = list_to_comma_str([*range(len(devices))])
        device_num = get_ans('Select one target port ({}): '.format(
            dev_comma_sep), [*range(len(devices))])
        device_info.num = int(device_num)
        ser.port = devices[device_info.num]

    try:
        ser.open()
        return ser
    except Exception as e:
        show_window(
            str(type(e)), 'Error: occurs when opening serial', exit_=True)
        raise TrackingError('Error: error occurs when opening serial')


def print_devices(devices: List[str]) -> None:
    """Print devices"""
    for idx, device in enumerate(devices):
        print('%3d: open %s' % (idx, device))


def select_cam_device_num() -> int:
    """Check cam devices and select"""
    print('[Camera Device]')
    print('Checking camera devices...')
    cam_list = get_cams_list()
    print('=>', cam_list)
    device_num = int(
        get_ans('Select one camera device ({}): '.format(
            list_to_comma_str(cam_list)),
            cam_list))
    return device_num


def select_options() -> Dict[str, bool]:
    """Select options"""
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

    if get_ans('Colorize video? (y/n): ') == 'y':
        color_video = True
    else:
        color_video = False
    return {'save_video': save_video,
            'save_csv': save_csv,
            'color_video': color_video}


def draw_circle(median_blur: cv2.medianBlur, mouse_info: MouseInfo, idx: int):
    """Draw a circle on a centroid of blured frame"""
    try:
        # put circle
        cv2.circle(median_blur,
                   (mouse_info.centerX[idx], mouse_info.centerY[idx]),
                   10, (150, 150, 150),  thickness=4)
    except Exception:
        # last point
        cv2.circle(median_blur,
                   (mouse_info.centerX[-1], mouse_info.centerY[-1]),
                   10, (150, 150, 150), thickness=4)


def release(cap_: cv2.VideoCapture, csv_: IO, avi_: cv2.VideoWriter) -> None:
    """GC"""
    cap_.release()
    if csv_ is not None:
        csv_.close()
    if avi_ is not None:
        avi_.release()

    cv2.destroyAllWindows()


def video_body(select_port: Serial, mouse_info: MouseInfo,
               camera_info: DeviceInfo, save_video: bool,
               save_csv: bool, frame_color: bool) -> None:
    """Process and output a video and logs"""
    cap = cv2.VideoCapture(camera_info.num)
    avifile = None
    csvfile = None
    nowtime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        avifile = cv2.VideoWriter(nowtime + '.avi', fourcc, 10.0, (1280, 480))

    if save_csv:
        csvfile = open(nowtime + '.csv', 'w')

    try:
        _video_body(cap, select_port, mouse_info,
                    avifile, csvfile, frame_color)
    except KeyboardInterrupt:
        print('SIGINT')
        release(cap, avifile, csvfile)
    except Exception as e:
        err, msg = type(e), str(e)
        release(cap, avifile, csvfile)
        show_window(
            '{}:{}'.format(err, msg), 'Error: occurs when processing', exit_=True)


def _video_body(cap: cv2.VideoCapture, select_port: Serial,
                mouse_info: MouseInfo, avifile: Optional[cv2.VideoWriter],
                csvfile: Optional[IO], frame_color: bool = False) -> None:
    """Helper"""
    # cap = cv2.VideoCapture(camera_info.num)
    kernel1 = np.ones((5, 5), np.uint8)  # kernel (size=5x5)
    kernel2 = np.ones((10, 10), np.uint8)  # kernel (size=10x10)

    t0 = time.perf_counter()
    idx = 0
    ret, frame = cap.read()
    while ret:
        idx += 1
        t1 = time.perf_counter()

        # image processing:
        # extract black(glaypix<5) segment from binarized image
        # black->white, others->black
        two = np.where(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                       < 5, 254, 0).astype('uint8')
        # morphology conversion($.closing.opening)
        cl = cv2.morphologyEx(two, cv2.MORPH_CLOSE, kernel1)
        op = cv2.morphologyEx(cl, cv2.MORPH_OPEN, kernel2)
        med_blur = cv2.medianBlur(op, 5)

        mouse_info.med_blur = med_blur
        info = mouse_info.center_operation(med_blur, idx)

        info_str = str(info)
        select_port.write(info_str.encode('utf-8'))
        select_port.write(b'\n')

        timestamp = datetime.now().strftime(
            '%Y-%m-%d_%H:%M:%S.%f')[:-5]

        # infos := "${ },${elapsed_sec/10},${timestamp}"
        infos = '{},{},{}'.format(
            info_str[0:6], int((t1-t0)/10), timestamp)
        select_port.reset_output_buffer()

        # print timestamp
        print(infos)

        # pastes timestamp on upper of frame
        cv2.putText(frame, infos, (40, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), thickness=2)
        draw_circle(med_blur, mouse_info, idx)

        if csvfile is not None:
            csvfile.write(infos)
            csvfile.write('\n')

        if frame_color:
            med_blur = cv2.cvtColor(med_blur, cv2.COLOR_GRAY2BGR)
            side_by_side = np.hstack([med_blur, frame])
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            side_by_side = cv2.cvtColor(
                cv2.hconcat([med_blur, frame]), cv2.COLOR_GRAY2BGR)
        if avifile is not None:
            avifile.write(side_by_side)

        cv2.imshow('frames', side_by_side)

        t2 = time.perf_counter()
        if cv2.waitKey(1) & 0xFF == ord('q') or t2 - t1 > 25200:
            break

        ret, frame = cap.read()

    release(cap, avifile, csvfile)


def main() -> None:
    """Main"""
    print(BANNER)

    s = select_port()
    m = MouseInfo()
    m.centerX = [0, 0]
    m.centerY = [0, 0]
    m.centlist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    c = DeviceInfo()
    c.num = select_cam_device_num()

    options = select_options()

    show_window('info', 'If you quit, type "q" on cam window'
                        '\nor "Ctrl+C" on terminal.')

    video_body(s, m, c, options['save_video'],
               options['save_csv'], options['color_video'])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('SIGINT')
        exit(0)
