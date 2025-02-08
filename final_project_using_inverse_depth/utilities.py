import numpy as np
import cv2
import readline
import torch
import matplotlib.pyplot as plt
from collections import namedtuple
import os
import csv
from glob import glob
from copy import deepcopy
from tqdm import tqdm
import random
import cv2

Gt = namedtuple('Gt', ['K', 'R', 'T'])

eps = 1e-15

def ReadCovisibilityData(filename):
    covisibility_dict = {}
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue
            covisibility_dict[row[1]] = float(row[2])  # the 1st column is the df index

    return covisibility_dict


def NormalizeKeypoints(keypoints, K):
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])
    return keypoints


def ComputeEssentialMatrix(F, K1, K2, kp1, kp2):
    '''Compute the Essential matrix from the Fundamental matrix, given the calibration matrices. Note that we ask participants to estimate F, i.e., without relying on known intrinsics.'''

    # Warning! Old versions of OpenCV's RANSAC could return multiple F matrices, encoded as a single matrix size 6x3 or 9x3, rather than 3x3.
    # We do not account for this here, as the modern RANSACs do not do this:
    # https://opencv.org/evaluating-opencvs-new-ransacs
    assert F.shape[0] == 3, 'Malformed F?'

    # Use OpenCV's recoverPose to solve the cheirality check:
    # https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#gadb7d2dfcc184c1d2f496d8639f4371c0
    E = np.matmul(np.matmul(K2.T, F), K1).astype(np.float64)

    kp1n = NormalizeKeypoints(kp1, K1)
    kp2n = NormalizeKeypoints(kp2, K2)
    num_inliers, R, T, mask = cv2.recoverPose(E, kp1n, kp2n)

    return E, R, T


def ArrayFromCvKps(kps):
    '''Convenience function to convert OpenCV keypoints into a simple numpy array.'''

    return np.array([kp.pt for kp in kps])


def QuaternionFromMatrix(matrix):
    '''Transform a rotation matrix into a quaternion.'''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                  [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                  [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                  [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
    K /= 3.0

    # The quaternion is the eigenvector of K that corresponds to the largest eigenvalue.
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0:
        np.negative(q, q)

    return q


def ExtractSiftFeatures(image, detector, num_features):
    '''Compute SIFT features for a given image.'''

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kp, desc = detector.detectAndCompute(gray, None)
    return kp[:num_features], desc[:num_features]


def ComputeErrorForOneExample(q_gt, T_gt, q, T, scale):
    '''Compute the error metric for a single example.

    The function returns two errors, over rotation and translation. These are combined at different thresholds by ComputeMaa in order to compute the mean Average Accuracy.'''

    q_gt_norm = q_gt / (np.linalg.norm(q_gt) + eps)
    q_norm = q / (np.linalg.norm(q) + eps)

    loss_q = np.maximum(eps, (1.0 - np.sum(q_norm * q_gt_norm) ** 2))
    err_q = np.arccos(1 - 2 * loss_q)

    # Apply the scaling factor for this scene.
    T_gt_scaled = T_gt * scale
    T_scaled = T * np.linalg.norm(T_gt) * scale / (np.linalg.norm(T) + eps)

    err_t = min(np.linalg.norm(T_gt_scaled - T_scaled), np.linalg.norm(T_gt_scaled + T_scaled))

    return err_q * 180 / np.pi, err_t


def ComputeMaa(err_q, err_t, thresholds_q, thresholds_t):
    '''Compute the mean Average Accuracy at different tresholds, for one scene.'''

    assert len(err_q) == len(err_t)

    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(thresholds_q, thresholds_t):
        acc += [(np.bitwise_and(np.array(err_q) < th_q, np.array(err_t) < th_t)).sum() / len(err_q)]
        acc_q += [(np.array(err_q) < th_q).sum() / len(err_q)]
        acc_t += [(np.array(err_t) < th_t).sum() / len(err_t)]
    return np.mean(acc), np.array(acc), np.array(acc_q), np.array(acc_t)


def BuildCompositeImage(im1, im2, axis=1, margin=0, background=1):
    '''Convenience function to stack two images with different sizes.'''

    if background != 0 and background != 1:
        background = 1
    if axis != 0 and axis != 1:
        raise RuntimeError('Axis must be 0 (vertical) or 1 (horizontal')

    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape

    if axis == 1:
        composite = np.zeros((max(h1, h2), w1 + w2 + margin, 3), dtype=np.uint8) + 255 * background
        if h1 > h2:
            voff1, voff2 = 0, (h1 - h2) // 2
        else:
            voff1, voff2 = (h2 - h1) // 2, 0
        hoff1, hoff2 = 0, w1 + margin
    else:
        composite = np.zeros((h1 + h2 + margin, max(w1, w2), 3), dtype=np.uint8) + 255 * background
        if w1 > w2:
            hoff1, hoff2 = 0, (w1 - w2) // 2
        else:
            hoff1, hoff2 = (w2 - w1) // 2, 0
        voff1, voff2 = 0, h1 + margin
    composite[voff1:voff1 + h1, hoff1:hoff1 + w1, :] = im1
    composite[voff2:voff2 + h2, hoff2:hoff2 + w2, :] = im2

    return (composite, (voff1, voff2), (hoff1, hoff2))


def DrawMatches(im1, im2, kp1, kp2, matches, axis=1, margin=0, background=0, linewidth=2):
    '''Draw keypoints and matches.'''

    composite, v_offset, h_offset = BuildCompositeImage(im1, im2, axis, margin, background)

    # Draw all keypoints.
    for coord_a, coord_b in zip(kp1, kp2):
        composite = cv2.drawMarker(composite, (int(coord_a[0] + h_offset[0]), int(coord_a[1] + v_offset[0])),
                                   color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)
        composite = cv2.drawMarker(composite, (int(coord_b[0] + h_offset[1]), int(coord_b[1] + v_offset[1])),
                                   color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)

    # Draw matches, and highlight keypoints used in matches.
    for idx_a, idx_b in matches:
        composite = cv2.drawMarker(composite, (int(kp1[idx_a, 0] + h_offset[0]), int(kp1[idx_a, 1] + v_offset[0])),
                                   color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
        composite = cv2.drawMarker(composite, (int(kp2[idx_b, 0] + h_offset[1]), int(kp2[idx_b, 1] + v_offset[1])),
                                   color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
        composite = cv2.line(composite,
                             tuple([int(kp1[idx_a][0] + h_offset[0]),
                                    int(kp1[idx_a][1] + v_offset[0])]),
                             tuple([int(kp2[idx_b][0] + h_offset[1]),
                                    int(kp2[idx_b][1] + v_offset[1])]), color=(0, 0, 255), thickness=1)
    return composite


def LoadCalibration(filename):
    '''Load calibration data (ground truth) from the csv file.'''

    calib_dict = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue

            camera_id = row[1]
            K = np.array([float(v) for v in row[2].split(' ')]).reshape([3, 3])
            R = np.array([float(v) for v in row[3].split(' ')]).reshape([3, 3])
            T = np.array([float(v) for v in row[4].split(' ')])
            calib_dict[camera_id] = Gt(K=K, R=R, T=T)

    return calib_dict


def read_all_images(src, scene):
    images_dict = {}
    for filename in glob(f'{src}/{scene}/images/*.jpg'):
        cur_id = os.path.basename(os.path.splitext(filename)[0])

        # OpenCV expects BGR, but the images are encoded in standard RGB, so you need to do color conversion if you use OpenCV for I/O.
        images_dict[cur_id] = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

    print(f'Loaded {len(images_dict)} images.')


def triangulate_points(K1, K2, R1, R2, T1, T2, points1, points2):
    """
    Triangulates multiple 3D points given corresponding 2D points in two images.

    Parameters:
        K1, K2: (3x3) Camera intrinsic matrices.
        R1, R2: (3x3) Rotation matrices for Camera 1 and Camera 2.
        T1, T2: (3x1) Translation vectors for Camera 1 and Camera 2.
        points1: (Nx2) Array of (x, y) coordinates in Image 1.
        points2: (Nx2) Array of (x, y) coordinates in Image 2.

    Returns:
        points_3D:  (Nx3) Array of 3D points in the world coordinate system.
        points3d_1: (Nx3) Array of 3D points in the first camera coordinate system.
    """

    # Construct Projection Matrices for both cameras
    P1 = K1 @ np.hstack((R1, T1.reshape(3, 1)))  # Projection matrix for Camera 1
    P2 = K2 @ np.hstack((R2, T2.reshape(3, 1)))  # Projection matrix for Camera 2

    # Convert points to homogeneous coordinates (required for cv2.triangulatePoints)
    points1_hom = np.vstack((points1.T, np.ones((1, points1.shape[0]))))
    points2_hom = np.vstack((points2.T, np.ones((1, points2.shape[0]))))

    # Triangulate points (result is in homogeneous coordinates)
    points_4D = cv2.triangulatePoints(P1, P2, points1_hom[:2], points2_hom[:2])

    # Convert from homogeneous (4D) to Cartesian (3D)
    points_3D = (points_4D[:3] / points_4D[3]).T  # Transpose to get Nx3 shape

    points_3D_1 = (R1 @ points_3D.T) + T1.reshape(3, 1)  # Transpose to get Nx3 shape

    return points_3D, points_3D_1


def reproject_points(K, R, T, points_3D):
    """
    Projects 3D world points back into a camera's image plane.

    Returns:
        projected_points: (Nx2) array of reprojected 2D points.
    """
    # Transform 3D points from World to Camera
    points_3D_cam2 = (R @ points_3D.T) + T.reshape(3, 1)  # Shape (3, N)

    # Project onto the image plane
    points_2D_hom = K @ points_3D_cam2  # Shape (3, N)

    # Normalize homogeneous coordinates
    points_2D = (points_2D_hom[:2] / points_2D_hom[2]).T  # Shape (N, 2)

    return points_2D, points_3D_cam2

def project_points(pts1, depth, K1, K2, R1, R2, t1, t2):
    """
    Given 2D points in image I1, project them into I2 using known camera parameters.

    Args:
        pts1: (N, 2) array of points in image1.
        K1, K2: Intrinsic matrices of camera1 and camera2.
        R1, R2: Rotation and translation of camera1.
        t1, t2: Rotation and translation of camera2. 3x1

    Returns:
        pts2: (N, 2) array of corresponding points in image2.
    """

    # Convert 2D points to homogeneous coordinates
    pts1_h = np.concatenate([pts1, np.ones((pts1.shape[0], 1))], axis=1).T  # (3, N)

    # Compute Projection Matrices
    # P1 = K1 @ np.hstack((R1, t1))  # Projection matrix for camera1
    P2 = K2 @ np.hstack((R2, t2))  # Projection matrix for camera2

    # Convert image points to normalized camera coordinates
    pts1_cam = np.linalg.inv(K1) @ pts1_h  # (3, N)

    # Assume a depth (since we don't have true depth, set arbitrary scale factor)
    pts3D = R1.T @ (pts1_cam * depth - t1)  # Back-project into 3D world

    # Project 3D points into image2
    pts2_h = P2 @ np.vstack((pts3D, np.ones((1, pts3D.shape[1]))))  # (3, N)

    # Convert back to 2D by normalizing homogeneous coordinates
    pts2 = (pts2_h[:2] / pts2_h[2]).T  # Convert to (N, 2)

    return pts2


def compute_homography(K1, K2, R1, t1, R2, t2, n, d):
    """
    Computes the homography matrix H given two camera poses and a plane in the scene.

    Args:
        K1, K2: Intrinsic matrices of camera1 and camera2.
        R1, t1: Rotation and translation of camera1.
        R2, t2: Rotation and translation of camera2.
        n: Normal vector of the reference plane in world coordinates.
        d: Distance from the camera to the plane.

    Returns:
        H: 3x3 homography matrix.
    """
    # Compute relative rotation and translation
    R_rel = R2 @ np.linalg.inv(R1)
    t_rel = t2 - R_rel @ t1

    # Compute homography
    H = K2 @ (R_rel - (t_rel @ n.T) / d) @ np.linalg.inv(K1)

    return H


def transform_points_homography(pts1, H):
    """
    Applies homography transformation to a set of 2D points.

    Args:
        pts1: (N, 2) array of points in image1.
        H: 3x3 homography matrix.

    Returns:
        pts2: (N, 2) array of transformed points in image2.
    """
    # Convert to homogeneous coordinates
    pts1_h = np.concatenate([pts1, np.ones((pts1.shape[0], 1))], axis=1).T  # (3, N)

    # Apply homography transformation
    pts2_h = H @ pts1_h  # (3, N)

    # Convert back to 2D by normalizing homogeneous coordinates
    pts2 = (pts2_h[:2] / pts2_h[2]).T  # (N, 2)

    return pts2


# img_path = 'train/british_museum/images/01858319_78150445.jpg'
def get_midas_depth(img_path, midas, show_flag=False):
    # Load the MiDaS model from the Intel-ISL/MiDaS GitHub repository via Torch Hub

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    input_batch = transform(img_rgb)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch = input_batch.to(device)

    with torch.no_grad():
        inv_depth = midas(input_batch)
        inv_depth = torch.nn.functional.interpolate(inv_depth.unsqueeze(1),  size=img_rgb.shape[:2],
                                                     mode="bicubic",    align_corners=False,).squeeze()


    if show_flag:
        plt.figure()
        plt.subplot(121)
        plt.imshow(img_rgb)

        plt.subplot(122)
        plt.imshow(inv_depth.cpu().numpy())
        plt.colorbar()
        plt.savefig('depth.png', bbox_inches ="tight")
        plt.show()

    return img_rgb, inv_depth

def warp_image(image1, image2, depth1, K1, K2, R1, R2, T1, T2):
    """ Warp image1 using depth1 to the viewpoint of image2. """
    h, w = image1.shape[:2]
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    pixels = np.stack((x.flatten(), y.flatten(), np.ones_like(x.flatten())), axis=0)  # Homogeneous coords

    # Convert pixels to 3D camera coordinates
    points_3D = np.linalg.inv(K1) @ (pixels * depth1.flatten())

    # Transform from Camera 1 to Camera 2
    points_3D_cam2 = R2 @ (R1.T @ (points_3D - T1.reshape(3, 1))) + T2.reshape(3, 1)

    # Project onto Image 2 plane
    pixels_2_hom = K2 @ points_3D_cam2
    pixels_2 = (pixels_2_hom[:2] / pixels_2_hom[2]).reshape(2, h, w)  # Normalize

    # Warp image using remapping
    # remap() for every pixel in the destination image,
    # lookup where it comes from in the source image, and then assigns an interpolated value.
    warped_image2 = cv2.remap(image2, pixels_2[0].astype(np.float32), pixels_2[1].astype(np.float32), cv2.INTER_LINEAR)

    return warped_image2


def show_history():
    # getting history of the python session:
    for i in range(readline.get_current_history_length()):
        print(readline.get_history_item(i + 1))






