import numpy as np
from numpy.core.numeric import identity
from numpy.linalg import norm



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


def compute_RTd(norm_vec, obj_ref_X, obj_mir):
    '''
    Params:
        norm_vec: (M, 3)
        obj_ref_X: (N, 3)
        obj_mir: (M, N, 3)
    '''
    A = np.zeros((27, 12))

    A[:, :3] = np.tile(np.eye(3), (9, 1))

    A[0:9, 3] = 2 * np.tile(norm_vec[0], 3)
    A[9:18, 4] = 2 * np.tile(norm_vec[1], 3)
    A[18:27, 5] = 2 * np.tile(norm_vec[2], 3)
    
    xy = np.repeat(obj_ref_X[:, :2], 3, axis=0)
    xy = np.repeat(xy, 3, axis=1)

    A[:, 6:] = np.tile(xy, (3, 1)) * np.tile(np.eye(3), (9, 2))

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
    T = Z[:3]
    d = Z[3: 6]

    return R, T, d


def main():
    # reference object
    obj_ref_X = np.array((
        (0., 0., 0.),
        (255., 100., 0.),
        (10., 150., 0.),
    )) # (N, 3)

    obj_ref = rot_x(15/180*np.pi, obj_ref_X.T).T
    obj_ref += np.array((150., 100., 10.))
    print("Reference object in Camera coordinate:")
    print(obj_ref)

    # 3 predefined mirror poses
    norm_vec = np.array((
        (0., 0., -1),
        (1., 0., -1.),
        (-1., 1., -1.)
    )) # (M, 3)
    
    norm_vec /= np.linalg.norm(norm_vec, axis=1, keepdims=True)
    print("\nGroundtruth norm vectors:")
    print(norm_vec)
    mirror_distance = np.array((200., 300., 300.)).reshape(-1, 1) # (M, 1)

    obj_mir = forward(obj_ref, norm_vec, mirror_distance)

    mij = np.cross(norm_vec, np.roll(norm_vec, -1, axis=0))
    print('\nGroundtruth mij')
    print(mij)

    est_mij = backward(obj_mir, mij)
    print("\nEstimated mij")
    print(est_mij)

    norm_vec = axis_to_norm(est_mij)
    print("\nNorm vector")
    print(norm_vec)

    R, T, d = compute_RTd(norm_vec, obj_ref_X, obj_mir)

    print("\nResult:")
    print("R:")
    print(R)
    print("T:")
    print(T)
    print("d:")
    print(d)
    print("\nEstimated reference object in Camera corrdinate:")
    print((R@(obj_ref_X.T)).T + T)

    


if __name__ == "__main__":
    main()


