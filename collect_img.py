from os import path
import os
import cv2
import numpy as np
import pickle
import yaml
import glob


def create_checkboard(monitor_res, monitor_size):
    '''
    Params:
        monitor_res: (width, height)
            size of monitor in pixel
        monitor_size: (width, height)
            size of monitor in mm
    '''
    board_res = monitor_res[0]//4*3, monitor_res[1]
    board_size = monitor_size[0]//4*3, monitor_size[1]
    
    width, heigh = board_res
    square_pixel = min(heigh, width) // 6
    checkboard = np.zeros((heigh, width), dtype=np.uint8)
    n_cols  = width // square_pixel
    n_rows = heigh // square_pixel
    vertice_shape = (n_cols-3, n_rows-3)

    pixel_size = (
        board_size[0]/board_res[0], 
        board_size[1]/board_res[1], 
        0
    )

    objp = np.zeros(
        (vertice_shape[0]*vertice_shape[1], 3), 
        np.float32
    )
    objp[:,:2] = np.mgrid[2:2+vertice_shape[0], 2:2+vertice_shape[1]].T.reshape(-1, 2)
    objp *= square_pixel
    objp *= pixel_size
    objp = objp.reshape((*vertice_shape, 3))
   
    for i in range(n_rows):
        for j in range(n_cols):
            color = ((i+j) % 2) * 255
            if i == 0 or i == n_rows-1 or j == 0 or j == n_cols-1:
                color = 255            

            xmin = i * square_pixel
            xmax = xmin + square_pixel
            ymin = j * square_pixel
            ymax = ymin + square_pixel
            checkboard[xmin:xmax, ymin:ymax] = color

    return checkboard, objp



def main(camcap, img_root, monitor_res, monitor_size):
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("camera", cv2.WND_PROP_TOPMOST, 1)

    # check window flags at https://docs.opencv.org/4.5.3/d0/d90/group__highgui__window__flags.html#gaeedf4023e777f896ba6b9ffb156f57b8
    cv2.namedWindow("full window", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("full window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty("full window", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)

    output = dict()
    
    img, objp = create_checkboard(monitor_res, monitor_size)
    output['objp'] = objp
    output['mon_res'] = monitor_res
    output['mon_size'] = monitor_size
    
    pickle.dump(
        output,
        open(path.join(img_root, "checkboard_monitor.pkl"), "wb")
    )
    cv2.imwrite(f"{img_root}/rendered_board.jpg", img)
    
    cv2.imshow("full window", img)

    
    for fname in glob.glob(path.join(img_root, '*.png')):
        os.remove(fname)

    n = 0

    ret, frame = camcap.read()

    while(ret):
        ret, frame = camcap.read()

        cv2.imshow('camera', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'{img_root}/{n}.png', frame)
            n += 1
        
    cv2.destroyAllWindows()


if __name__ == "__main__":

    with open("hardware_specs.yaml", "r") as stream:
        data = yaml.safe_load(stream)
        mon_dict = data["monitor"]
        cam_dict = data["camera"]

    # mon_dict = {
    #     'G5': ((1920, 1080), (344, 194)), # width, height
    #     'Dell24': ((1920, 1080), (527, 296)),
    #     'Dell27': ((1920, 1080), (598, 336)),
    #     'Dell274K': ((1920, 1080), (610, 350))
        
    # }

    monitor_name = input("Monitor name: ")
    camera_name = input("Camera name: ")

    img_root = f'data/extrinsic/mon{monitor_name}_cam{camera_name}'
    if not path.isdir(img_root):
        os.mkdir(img_root)
    
    monitor_res, monitor_size = mon_dict[monitor_name]  # pixel, mm

    # define a video capture object
    camcap = cv2.VideoCapture(4)
    camcap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_dict[camera_name][0])
    camcap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_dict[camera_name][1])

    main(camcap, img_root, monitor_res, monitor_size)

    camcap.release()
