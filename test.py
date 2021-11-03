import numpy as np
import cv2


def rot_x(alpha, points):
    R = np.array((
        (1, 0, 0),
        (0, np.cos(alpha), -np.sin(alpha)),
        (0, np.sin(alpha), np.cos(alpha))
    ))
    return np.matmul(R, points)

def rot_y(alpha, points):
    R = np.array((
        (np.cos(alpha), 0, np.sin(alpha)),
        (0, 1, 0),
        (-np.sin(alpha), 0, np.cos(alpha))
    ))
    return np.matmul(R, points)



def main():
    points_3d = np.array((
        (1000, 0, 0),
        (1000, 1000, 0),
        (1000, 1000, 1000),
        (800, 0, 1000)
    ))

    points_3d = rot_x(np.pi/2, points_3d.T).T
    points_3d += np.array((10., 5., 1400.))


    intrinsic = np.array([
        [641.00199769,   0.        , 320.40306364],
        [  0.        , 641.02737901, 254.34007326],
        [  0.        ,   0.        ,   1.        ]
    ])
    dist = np.array([[ 0., 0,  0.,  0.,  0.]])

    points_2d = np.matmul(intrinsic, points_3d.T)
    points_2d /= points_2d[2]
    points_2d = points_2d[:2].T
    points_2d = np.expand_dims(points_2d, axis=1)

    print(points_2d)
    
    ret, rvec, tvec = cv2.solvePnP(
        points_3d,
        points_2d,
        intrinsic,
        dist,
        # useExtrinsicGuess=True,
        # flags=cv2.SOLVEPNP_ITERATIVE
    )

    
    # modelAxes = np.array([
    #     np.array((car_x[0, 0], car_y[0, 0], car_z)).reshape(1, 3)
    #     # np.array((car_x[0], car_y[0], car_z)).reshape(1, 3)
        
    # ]) # (1, 1, 3)

    # project 3D point on front image 
    projAxes, _ = cv2.projectPoints(
        points_3d, 
        rvec,
        tvec,
        intrinsic, 
        None
    )
    print(projAxes)

if __name__ == "__main__":
    main()
