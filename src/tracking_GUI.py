import getpass
import os
import sys
import time
import warnings
from datetime import datetime
from typing import IO, Any, List, NewType, Optional, Tuple, TypedDict, Union

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
v0.14
'''


class DeviceInfo():
    def __init__(self):
        self.num: int = 0


class TrackingError(Exception):
    pass


class MouseInfo():
    def init(self):
        self.binarized_frame = np.ndarray((2,))
        self.centerX = []
        self.centerY = []
        self.centlist = []

    def center_operation(self, binarized_frame: np.ndarray, idx: int) -> float:
        """Calculate the amount of movement of the center of gravity
        Measure the movement of the center of gravity from the previous frame.
        """
        _, _, stats, centroids = cv2.connectedComponentsWithStats(
            binarized_frame, 4)
        stats = stats[1:]  # Remove information from the entire image(idx=0)
        # Sort by size
        # Exclusion of obviously different shapes by mice
        # a_,b_=sorted([w,h]);OK if a_/b_>=0.1
        for _ in range(len(stats)):
            a_, b_ = np.sort(stats[_, 2:4])
            if a_/b_ < 0.3:
                stats[_, -1] = 0
        try:
            # Select the one with the largest size
            if len(stats) > 0:
                max_idx = np.argmax(stats[:, -1]) + 1
                # substitute useless components from frame (too slow):
                # for _ in range(1, nlabels):
                #     if _ != max_idx:
                #         binarized_frame[labels == _] = 0
                cx = int(centroids[max_idx][0])
                cy = int(centroids[max_idx][1])
            else:
                cx, cy = self.centerX[-1], self.centerY[-1]

            prev_cx = self.centerX[-1]
            prev_cy = self.centerY[-1]

            self.centerX.append(cx)
            self.centerY.append(cy)
            # measure distance between prev and current centroids
            self.centlist.append((prev_cx - cx)**2 + (prev_cy - cy)**2)

        except ValueError:
            self.centlist.append(0)

        # returns: 300 * z-value
        # zvalue is alse called standard score
        z = (self.centlist[idx] - np.mean(self.centlist))/np.std(self.centlist)
        return 300 * np.abs(z)


class Options(TypedDict):
    save_video: bool
    save_csv: bool
    color_video: bool


def select_options() -> Options:
    """Select options"""
    print('[Options]')
    # set if save:
    save_video = get_ans('Save video(.avi) ? (y/n): ') == 'y'
    save_csv = get_ans('Save result(.csv)? (y/n): ') == 'y'
    color_video = get_ans('Colorize video? (y/n): ') == 'y'

    return {'save_video': save_video,
            'save_csv': save_csv,
            'color_video': color_video}


def get_ans(question: str, selections: Union[List[str], List[int]] = ['y', 'n']):
    """Question and receive a selection
    """
    reply = input(question)
    selections = list(map(str,  selections))
    if reply in selections:
        return reply
    else:
        return get_ans('invalid answer. retry: ', selections)


def get_cams_list() -> List[int]:
    """Get a list of camera devices
    """
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
    if exit_:
        exit(1)


def select_port() -> Serial:
    """Check port devices with serial communication and select"""
    ser: Serial = serial.Serial()
    ser.baudrate = 9600  # same as Serial.begin in Arduino
    ser.timeout = 0.1
    print('[Serial Device]')
    print('Checking serial devices... (baudrate: {}, timeout: {})'.format(
        ser.baudrate, ser.timeout))

    ports = list_ports.comports()  # get port data
    devices: List[str] = [info.device for info in ports]
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


def draw_circle(binarized_frame: np.ndarray, mouse_info: MouseInfo) -> None:
    """Draw a circle on a centroid of blured frame"""
    binarized_frame = cv2.circle(binarized_frame,
                                 (mouse_info.centerX[-1],
                                  mouse_info.centerY[-1]),
                                 10, (150, 150, 150),  thickness=4)


def release(cap_: Optional[cv2.VideoCapture] = None,
            csv_: Optional[IO] = None, avi_: Optional[cv2.VideoWriter] = None) -> None:
    """GC"""
    if cap_ is not None:
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
    cap = cv2.VideoCapture(sys.argv[1]
                           if len(sys.argv) == 2 and os.path.isfile(sys.argv[1])
                           else camera_info.num)
    avifile = None
    csvfile = None
    nowtime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')

    if save_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))*2
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        avifile = cv2.VideoWriter(
            nowtime + '.avi', fourcc, fps, (width, height))

    if save_csv:
        csvfile = open(nowtime + '.csv', 'w')

    try:
        _video_body(cap, select_port, mouse_info,
                    avifile, csvfile, frame_color)
    except KeyboardInterrupt:
        print('SIGINT')
        release(cap, csvfile, avifile)


def binarize(frame: np.ndarray,
             range_low: Tuple[int, int, int] = (0, 0, 0, ),
             range_up: Tuple[int, int, int] = (45, 255, 23,)) -> np.ndarray:
    """image processing (new):
    remove green cable
    """
    def make_hsv_range(low: Tuple[int, int, int], up: Tuple[int, int, int]
                       ) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(low), np.array(up)
    filter_ = make_hsv_range(range_low, range_up)
    # filter and remove color except a color of black mouse
    mask = cv2.inRange(frame, *filter_)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((20, 20), np.uint8))
    mask = cv2.dilate(mask, np.ones((50, 50), np.uint8))

    return mask


def resize_frame(frame: np.ndarray, ratio: float = 0.5) -> np.ndarray:
    h, w = frame.shape[0:2]
    return cv2.resize(frame, (int(w*ratio), int(h*ratio)))


def _video_body(cap: cv2.VideoCapture, select_port: Serial,
                mouse_info: MouseInfo, avifile: Optional[cv2.VideoWriter],
                csvfile: Optional[IO], frame_color: bool = False) -> None:
    """Helper"""

    t0 = time.perf_counter()
    idx = 0
    ret, frame = cap.read()
    while ret:
        idx += 1
        t1 = time.perf_counter()
        binarized_frame = binarize(frame)

        mouse_info.binarized_frame = binarized_frame
        info = mouse_info.center_operation(binarized_frame, idx)

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
        draw_circle(binarized_frame, mouse_info)

        if csvfile is not None:
            csvfile.write(infos)
            csvfile.write('\n')

        if frame_color:
            side_by_side = np.hstack(
                [cv2.cvtColor(binarized_frame, cv2.COLOR_GRAY2BGR), frame])
        else:
            mono = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            side_by_side = np.hstack([binarized_frame, mono])
        if avifile is not None:
            avifile.write(side_by_side)

        if os.environ.get('TEST') == '1':
            ret, frame = cap.read()
            continue

        cv2.imshow('frames', resize_frame(side_by_side))

        t2 = time.perf_counter()
        if cv2.waitKey(1) & 0xFF == ord('q') or t2 - t1 > 25200:
            break

        ret, frame = cap.read()

    release(cap, csvfile, avifile)


def main() -> None:
    """Main"""
    print(BANNER)

    s = select_port()
    m = MouseInfo()
    m.centerX = [0, 0]
    m.centerY = [0, 0]
    m.centlist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    c = DeviceInfo()
    if len(sys.argv) < 2:
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
