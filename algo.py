import numpy as np


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


def distance_points_planes(P: np.ndarray, 
                         norm_vec: np.ndarray, 
                         d: np.ndarray) -> np.ndarray:
    '''
    Params:
        P: (N, 3)
            batch of points
        norm_vec: (M, 3)
            batch of normal unit vectors of planes
        d: (M, 1)
            batch of distances from origin to planes
    Returns:
        distance: (M, N)
    '''
    distance = np.matmul(norm_vec, P.T) + d
    distance = np.abs(distance)
    return distance


def forward(obj_ref: np.ndarray, 
            norm_vec: np.ndarray, 
            mirror_distance: np.ndarray) -> np.ndarray:
    '''
    Compute mirrored object of referenc object via mirror's poses
    '''
    distance = distance_points_planes(obj_ref, norm_vec, mirror_distance) # (M, N)

    obj_mir = np.expand_dims(obj_ref, axis=0) - 2 * np.expand_dims(distance, axis=-1) * np.expand_dims(norm_vec, axis=-2) # (M, N, 3)

    return obj_mir
    

def axis_to_norm(axis_vec):
    norm_vec = np.cross(
        axis_vec,
        np.roll(axis_vec, 1, axis=0)
    )
    norm_vec /= np.linalg.norm(norm_vec, axis=1, keepdims=True)

    # corect sign of z axis
    sign = np.where(norm_vec[:, 2] > 0, -1, 1).reshape(-1, 1)
    norm_vec *= sign

    return norm_vec


def backward(obj_mir: np.ndarray):
    Q = obj_mir - np.roll(obj_mir, -1, axis=0) #(M, N, 3)
    M = np.matmul(
        np.transpose(Q, (0, 2, 1)), 
        Q
    )
    e_val, e_vec = np.linalg.eig(M)
    idx_min = np.argmin(e_val, axis=1)
    e_vec = np.transpose(e_vec, (0, 2, 1))
    mij = e_vec[np.arange(len(idx_min)), idx_min]
    return mij  

def check_rotation_constrain(R):
    print(np.cross(R[:, 0], R[:, 1]))
    print(np.cross(R[:, 1], R[:, 2]))
    print(np.cross(R[:, 2], R[:, 0]))

    print(np.linalg.norm(R[:, 0]))
    print(np.linalg.norm(R[:, 1]))
    print(np.linalg.norm(R[:, 2]))

    return

def compute_RTd(norm_vec, obj_ref_X, obj_mir):
    '''
    Params:
        norm_vec: (M, 3)
        obj_ref_X: (N, 3)
        obj_mir: (M, N, 3)
    '''
    n_point = obj_ref_X.shape[0]
    A = np.zeros((9*n_point, 12))

    A[:, :3] = np.tile(np.eye(3), (3*n_point, 1))

    A[0:3*n_point, 3] = 2 * np.tile(norm_vec[0], n_point)
    A[3*n_point:6*n_point, 4] = 2 * np.tile(norm_vec[1], n_point)
    A[6*n_point:9*n_point, 5] = 2 * np.tile(norm_vec[2], n_point)
    
    xy = np.repeat(obj_ref_X[:, :2], 3, axis=0)
    xy = np.repeat(xy, 3, axis=1)

    A[:, 6:] = np.tile(xy, (3, 1)) * np.tile(np.eye(3), (3*n_point, 2))

    B = -2 * np.expand_dims(norm_vec, 1) @ np.transpose(obj_mir, (0, 2, 1)) # (M, 1, N)

    B = B * np.expand_dims(norm_vec, -1) # (M, 3, N)

    B = np.transpose(B, (0, 2, 1)) # (M, N, 3)
    B += obj_mir # (M, N, 3)

    B = B.reshape(-1)
    
    Z = np.linalg.lstsq(A, B, rcond=None)[0]

    r1 = Z[-6:-3]
    r1 /= np.linalg.norm(r1)

    r2 = Z[-3:]
    r2 /= np.linalg.norm(r1)
    r3 = np.cross(r1, r2)
    r3 /= np.linalg.norm(r3)
    R = np.array((r1, r2, r3)).T

    # orthoganal procrustes prblem
    u, s, vh = np.linalg.svd(R)
    R = u @ vh
    
    t = Z[:3]
    d = Z[3: 6]

    return R, t, d


def create_synthesis_data():
    # reference object in screen coordinate
    obj_ref_X = np.array((
        (0., 0., 0.),
        (255., 100., 0.),
        (10., 150., 0.),
        (100., 255., 0.)
    )) # (N, 3)

    # reference object in camera coordinate
    obj_ref = rot_y(180/180*np.pi, obj_ref_X.T)
    obj_ref = rot_x(15/180*np.pi, obj_ref).T
    
    obj_ref += np.array((150., 10., 10.))
    print("Reference object in Camera coordinate:")
    print(obj_ref)

    # normal vectors of mirror poses
    norm_vec = np.array((
        (0., 0., -1),
        (0.5, 0., -1.),
        (-0.2, -0.2, -1.)
    )) # (M, 3)
    norm_vec /= np.linalg.norm(norm_vec, axis=1, keepdims=True)
    print("\nGroundtruth norm vectors:")
    print(norm_vec)

    # distance from camera to mirror poses
    mirror_distance = np.array((200., 300., 300.)).reshape(-1, 1) # (M, 1)

    # mirrored object in camera coordinate
    obj_mir = forward(obj_ref, norm_vec, mirror_distance)

    # groundtruth vectors lie on intersection of each couple of mirror poses
    mij = np.cross(norm_vec, np.roll(norm_vec, -1, axis=0))
    print('\nGroundtruth mij')
    print(mij)

    return obj_ref_X, obj_mir


def mirror_calib(obj_ref_X, obj_mir, debug=False):
    
    if debug:
        obj_ref_X, obj_mir = create_synthesis_data()

    # estimated vectors lie on intersection of each couple of mirror poses
    mij = backward(obj_mir)

    # convert axis vecotors to normal vectors
    norm_vec = axis_to_norm(mij)

    # compute transfomation between screen and camera 
    R, T, d = compute_RTd(norm_vec, obj_ref_X, obj_mir)

    if debug:
        print("\nEstimated mij")
        print(mij)
        print("\nNorm vector")
        print(norm_vec)
        print("\nResult:")
        print("R:")
        print(R)
        print("T:")
        print(T)
        print("d:")
        print(d)
        print("\nEstimated reference object in Camera corrdinate:")
        print((R@(obj_ref_X.T)).T + T)

    return R, T, d

    


if __name__ == "__main__":
    mirror_calib(None, None, debug=True)


