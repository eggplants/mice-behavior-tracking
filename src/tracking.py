import time
from datetime import datetime

import cv2
import numpy as np
import serial
from serial.tools import list_ports


# ===============================================================================
def select_port():
    ser = serial.Serial()
    ser.baudrate = 9600
    ser.timeout = 0.1       # タイムアウトの時間
    print("chaking serial", ser)  # natsu add

    ports = list_ports.comports()    # ポートデータを取得

    devices = [info.device for info in ports]

    if len(devices) == 0:
        # シリアル通信できるデバイスが見つからなかった場合
        print("error: device not found")
        return None
    elif len(devices) == 1:
        print("only found %s" % devices[0])
        ser.port = devices[0]
    else:
        # ポートが複数見つかった場合それらを表示し選択させる
        for i in range(len(devices)):
            print("%3d: open %s" % (i, devices[i]))
        print("input number of target port >> ", end="")
        num = int(input())
        ser.port = devices[num]

    # 開いてみる
    try:
        ser.open()
        return ser
    except Exception:
        print("error when opening serial")
        return None

    # -----connection_check--------------------------------------------------------------------------------------------------


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
    # ------------------------------------------------------------------------------------------------------

# prepare


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
        return 300*np.abs((self.centlist[i] - np.mean(self.centlist)) /
                          np.std(self.centlist))


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


def get_ans(question, selections=["y", "n"]):
    reply = input(question)
    selections = list(map(str,  selections))
    if reply in selections:
        return reply
    else:
        return get_ans("invalid answer. retry: ", selections)

# =============================================================================-


def video_body(ser, C):
    cams_list = [*range(3)]
    cap = cv2.VideoCapture(int(get_ans(
        'give me camera id ({}...): '.format(cams_list), cams_list)))
    if get_ans("video save?(y/n): ") == "y":
        video_save = True
    else:
        video_save = False

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    nowtime = str(datetime.now())
    filename = nowtime.replace(" ", "").replace(
        ".", "-").replace(":", "-") + str(".avi")
    csvfilename = nowtime.replace(" ", "").replace(
        ".", "-").replace(":", "-") + str(".csv")
    if video_save:
        out = cv2.VideoWriter(filename, fourcc,
                              10.0, (1280, 480))  # 1280,480

    kernel1 = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((10, 10), np.uint8)

    csvfile = open(csvfilename, "w")

    t0 = time.perf_counter()
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            i += 1
            t1 = time.perf_counter()
        # ========================画像処理=============================================================
            two = np.where(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                           < 5, 254, 0).astype('uint8')  # 5
            cl = cv2.morphologyEx(two, cv2.MORPH_CLOSE, kernel1)
            op = cv2.morphologyEx(cl, cv2.MORPH_OPEN, kernel2)
            mb = cv2.medianBlur(op, 5)
            # 数値処理
            C.mb = mb
            info = C.center_opelation(mb, i)

            info_str = str(info)
            ser.write(info_str.encode('utf-8'))
            ser.write(b'\n')

            timestamp = str(datetime.now())[0:21].replace(" ", "")
            infos = info_str[0:6] + "," + \
                str(int((t1-t0)/10)) + "," + timestamp
            ser.reset_output_buffer()
            # allinfo = info_str + "," + str(int((t1-t0)/10))  + "," +timestamp
            print(infos)
            csvfile.write(infos)
            csvfile.write("\n")

            cv2.putText(frame, infos, (40, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), thickness=2)
            try:
                # put round
                cv2.circle(mb, (C.centerX[i], C.centerY[i]),
                           10, (150, 150, 150),  thickness=4)
            except Exception:
                # last point
                cv2.circle(mb, (C.centerX[-1], C.centerY[-1]),
                           10, (150, 150, 150), thickness=4)

            out_frame_color = cv2.cvtColor(cv2.hconcat(
                [mb, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)]),
                cv2.COLOR_GRAY2BGR)

            if video_save:
                out.write(out_frame_color)

            # out.write(out_frame_color)
            cv2.imshow("frames", out_frame_color)

            t2 = time.perf_counter()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif t2 - t1 > 25200:
                break
            else:
                t2 = time.perf_counter()
                break

    csvfile.close()
    cap.release()
    if video_save:
        out.release()
    # out.release()
    cv2.destroyAllWindows()


# ---port open--- -----  ---------      ------------   ------    --------
ser = select_port()
# ---connection show-------- ---------  ----------- ------- ------- -----
# show_connection(ser)

# center 用クラス初期化
C = mouseinfo()
C.centerX = [0, 0]
C.centerY = [0, 0]
C.centlist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
video_body(ser, C)
