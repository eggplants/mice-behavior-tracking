import math
import statistics

from pynput import mouse  # please use v1.6.8

#
# This code might not run correctly! Use tracking_GUI.py
#

info = {
    'x_pos': [],
    'y_pos': [],
    'diffs': [],
    'filename': '',
}


def on_click(x, y, button, pressed):
    if pressed and button == mouse.Button.right:
        if len(info['x_pos']) == 0:
            info['x_pos'].append(x)
            info['y_pos'].append(y)
            print('Zero point was set')
        else:
            d = math.sqrt((info['x_pos'][-1] - x) **
                          2 + (info['y_pos'][-1] - y) ** 2)
            info['x_pos'].append(x)
            info['y_pos'].append(y)
            info['diffs'].append(d)
            print(f'{len(info["diffs"])}, {d:.3f}')
            if len(info['diffs']) % 10 == 0:
                return False


def save_file(info):
    with open('./' + info['filename'] + '.csv', 'w') as f:
        f.write('1,2,3,4,5,6,7,8,9,10,average\n')
        line_count = 0
        for i, d in enumerate(info['diffs']):
            f.write(f'{d:.3f},')
            if i % 10 == 9:
                i_start = line_count*10
                i_end = (line_count+1)*10
                d = info["diffs"][i_start:i_end]
                f.write(
                    f'{statistics.mean(d): .3f}\n')
                line_count += 1


def main():
    print('Click start position (Right-click)')
    while True:
        with mouse.Listener(on_click=on_click) as listener:
            listener.join()
        print('Want to save? Type file name and press Enter key.')
        info['filename'] = str(input())
        save_file(info)
        print(f'{info["filename"]}.csv is saved.\n')
        print('You can continue with right click.')
        info['x_pos'] = []
        info['y_pos'] = []


if __name__ == "__main__":
    main()
