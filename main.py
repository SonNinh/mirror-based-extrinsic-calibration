from collect_img import collect
from extrinsic_calib import calib as ext_calib
from twocams_calib import calib as bridge_calib 
import yaml
import os
from os import path
import cv2



def main(vertice_shape, cam_bridge_name, cam_target_name, monitor_name):
    with open("hardware_specs.yaml", "r") as stream:
        data = yaml.safe_load(stream)
        mon_dict = data["monitor"]
        cam_dict = data["camera"]


    cap_bridge = cv2.VideoCapture(cam_dict[cam_bridge_name]['index'])
    cap_bridge.set(cv2.CAP_PROP_FRAME_WIDTH, cam_dict[cam_bridge_name]['res'][0])
    cap_bridge.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_dict[cam_bridge_name]['res'][1])
    
    cap_target = cv2.VideoCapture(cam_dict[cam_target_name]['index'])
    cap_target.set(cv2.CAP_PROP_FRAME_WIDTH, cam_dict[cam_target_name]['res'][0])
    cap_target.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_dict[cam_target_name]['res'][1])
    
    # data collection for mirror-based calib
    img_root_bridge = f'data/extrinsic/mon{monitor_name}_cam{cam_bridge_name}'
    if not path.isdir(img_root_bridge):
        os.mkdir(img_root_bridge)
        print('Collecting data for mirror based calib ...')
        monitor_res = mon_dict[monitor_name]['res'] # pixel
        monitor_size = mon_dict[monitor_name]['size'] # mm
        collect(cap_bridge, img_root_bridge, monitor_res, monitor_size)
        print('Done!')


    # mirror-based calib
    print('Mirror-based calibrate ...')
    intrinsic_fname = f'data/intrinsic/cam{cam_bridge_name}/calib.pkl'
    ext_calib(img_root_bridge, intrinsic_fname)
    print('Done!')


    # extrinsic calibration between 2 cameras
    print('Bridge calibrate ...')
    img_root_target = f'data/extrinsic/mon{monitor_name}_cam{cam_target_name}'
    if not path.isdir(img_root_target):
        os.mkdir(img_root_target)
    
    bridge_calib(
        img_root_bridge, img_root_target, 
        vertice_shape, 
        cap_bridge, cap_target, 
        cam_bridge_name, cam_target_name,
        mon_dict[monitor_name]['res'], mon_dict[monitor_name]['size']
    )
    print('Done!')

    cap_target.release()
    cap_bridge.release()
    


if __name__ == '__main__':

    vertice_shape = (8, 6)
    cam_bridge_name = input("Bridge camera name: ")
    cam_target_name = input("Target camera name: ")
    monitor_name = input("Monitor name: ")

    main(vertice_shape, cam_bridge_name, cam_target_name, monitor_name)