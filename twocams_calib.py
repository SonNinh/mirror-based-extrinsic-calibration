import pickle
import cv2
import numpy as np
from visual import visual
import yaml
import os
from os import path


def correct_order(corners: np.ndarray,
                  vertice_shape):
    if corners[0, 0, 1] - corners[-1, 0, 1] > 0:
        corners = corners[::-1]
    
    corners = corners.reshape((vertice_shape[1], vertice_shape[0], 1, 2))
    corners = corners[:, ::-1]
    corners = corners.reshape((-1, 1, 2))
        
    return corners




def calib(img_root_bridge, img_root_target, vertice_shape, 
          cap_bridge, cap_target, 
          cam_bridge_name, cam_target_name,
          monitor_res, monitor_size
          ):

    cv2.namedWindow("bridge camera", cv2.WINDOW_NORMAL)
    cv2.moveWindow("bridge camera", 0, 0)
    cv2.namedWindow("target camera", cv2.WINDOW_NORMAL)
    cv2.moveWindow("target camera", monitor_res[0]//2, 0)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros(
        (vertice_shape[0]*vertice_shape[1], 3), 
        np.float32
    )
    objp[:,:2] = np.mgrid[:vertice_shape[0], :vertice_shape[1]].T.reshape(-1, 2)
    objp *= 29
    
    retA, frameA = cap_bridge.read()
    calibA = pickle.load(open(f'data/intrinsic/cam{cam_bridge_name}/calib.pkl', 'rb'))

    
    retB, frameB = cap_target.read()
    calibB = pickle.load(open(f'data/intrinsic/cam{cam_target_name}/calib.pkl', 'rb'))

    
    while retA and retB:
        cv2.imshow('bridge camera', frameA)
        cv2.imshow('target camera', frameB)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
                
            grayA = cv2.cvtColor(frameA, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(frameB, cv2.COLOR_BGR2GRAY)
            
            ret_board_A, cornersA = cv2.findChessboardCorners(grayA, vertice_shape, None)
            ret_board_B, cornersB = cv2.findChessboardCorners(grayB, vertice_shape, None)


            if ret_board_A and ret_board_B:
                
                cv2.destroyAllWindows()

                
                cornersA = cv2.cornerSubPix(grayA, cornersA, (11, 11), (-1,-1), criteria)
                cornersB = cv2.cornerSubPix(grayB, cornersB, (11, 11), (-1,-1), criteria)
                
                cornersA = correct_order(cornersA, vertice_shape)
                cornersB = correct_order(cornersB, vertice_shape)
                
                _, rvecA, tvecA = cv2.solvePnP(objp, cornersA, calibA['mtx'], calibA['dist'])
                _, rvecB, tvecB = cv2.solvePnP(objp, cornersB, calibB['mtx'], calibB['dist'])

                MA = np.zeros((4, 4))
                MA[:3, :3] = cv2.Rodrigues(rvecA)[0]
                MA[:3, 3:] = tvecA.reshape(-1, 1)
                MA[-1, -1] = 1

                MB = np.zeros((4, 4))
                MB[:3, :3] = cv2.Rodrigues(rvecB)[0]
                MB[:3, 3:] = tvecB.reshape(-1, 1)
                MB[-1, -1] = 1

                M_A_to_B = MB @ np.linalg.inv(MA)

                calib_mon_to_A = pickle.load(
                    open(path.join(img_root_bridge, "calib.pkl"), "rb")
                )
                M_mon_to_A = np.zeros((4, 4))
                M_mon_to_A[:3, :3] = calib_mon_to_A["R"]
                M_mon_to_A[:3, 3:] = calib_mon_to_A["t"].reshape(-1, 1)
                M_mon_to_A[-1, -1] = 1

                M_mon_to_B = M_A_to_B @ M_mon_to_A
                
                R = M_mon_to_B[:3, :3]
                t = M_mon_to_B[:3, 3]

                print("Rotation from monitor to target camera:")
                print(R)
                print("Translation from monitor to target camera:")
                print(t)

                output = dict()
                output["R"] = R
                output["t"] = t
                output["mon_res"] = monitor_res
                output["mon_size"] = monitor_size
                
                pickle.dump(
                    output, 
                    open(path.join(img_root_target, "calib.pkl"), "wb")
                )

                visual(
                    R, t, 
                    np.array(
                        (*monitor_size, 1), 
                        dtype=np.float32
                    )
                )

                break
        
        

        retA, frameA = cap_bridge.read()
        retB, frameB = cap_target.read()

    return


if __name__ == "__main__":

    vertice_shape = (8, 6)
    cam_bridge_name = input("Bridge camera name: ")
    cam_target_name = input("Traget camera name: ")
    monitor_name = input("Monitor name: ")

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

    img_root_bridge = f"data/extrinsic/mon{monitor_name}_cam{cam_bridge_name}"
    img_root_target = f'data/extrinsic/mon{monitor_name}_cam{cam_target_name}'
    if not path.isdir(img_root_target):
        os.mkdir(img_root_target)

    calib(
        img_root_bridge, img_root_target, 
        vertice_shape, 
        cap_bridge, cap_target, 
        cam_bridge_name, cam_target_name,
        mon_dict[monitor_name]['res'], mon_dict[monitor_name]['size']
    )

    cap_bridge.release()
    cap_target.release()