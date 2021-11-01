# import the opencv library
import cv2
import numpy as np


def create_checkboard(monitor_res, monitor_size):
    '''
    Params:
        monitor_res: (heigh, width)
            size of monitor in pixel
        monitor_size: (heigh, width)
            size of monitor in mm
    '''
    heigh, width = monitor_res
    square_pixel = min(heigh, width) // 4
    checkboard = np.zeros(monitor_res, dtype=np.uint8)
    n_rows = heigh // square_pixel
    n_cols  = width // square_pixel
    vertice_shape = (n_cols-1, n_rows-1)

    pixel_size = (
        monitor_size[1]/monitor_res[1], 
        monitor_size[0]/monitor_res[0], 
        0
    )

    objp = np.zeros(
        (vertice_shape[0]*vertice_shape[1], 3), 
        np.float32
    )
    objp[:,:2] = np.mgrid[1:1+vertice_shape[0], 1:1+vertice_shape[1]].T.reshape(-1, 2)
    objp *= square_pixel
    objp *= pixel_size
   
    for i in range(n_rows):
        for j in range(n_cols):
            color = ((i+j) % 2) * 255

            xmin = i * square_pixel
            xmax = xmin + square_pixel
            ymin = j * square_pixel
            ymax = ymin + square_pixel
            checkboard[xmin:xmax, ymin:ymax] = color

    return checkboard, vertice_shape



def main():
    cv2.namedWindow("full window", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("full window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    monitor_res = (1080, 1920) # pixel
    monitor_size = (194, 344) # mm

    img, vertice_shape = create_checkboard(monitor_res, monitor_size)
    cv2.imshow("full window", img)

    return
    # define a video capture object
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vid.set(cv2.CAP_PROP_FRAME_HEIGH, 480)
    n = 0

    ret, frame = vid.read()

    while(ret):
        ret, frame = vid.read()

        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'images/{n}.png', frame)
            n += 1
        

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()