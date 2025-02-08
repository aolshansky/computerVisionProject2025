import numpy as np
from fontTools.misc.bezierTools import epsilon
from sympy import false
from torch.backends import flags_frozen
import cv2
import utilities as utils
import math
import torch

from dpt_depth import get_dpt_depth, show_images, load_image

#
src = './project_data/train'
scene = "british_museum"


# --------------
calib_dict = utils.LoadCalibration(f'{src}/{scene}/calibration.csv')
covisibility_dict = utils.ReadCovisibilityData(f'{src}/{scene}/pair_covisibility.csv')

easy_subset = [k for k, v in covisibility_dict.items() if v >= 0.7]
difficult_subset = [k for k, v in covisibility_dict.items() if v >= 0.1 and v < 0.2]

pair = easy_subset[0]
id1, id2 = pair.split('-')

calib_1 = calib_dict[id1]
calib_2 = calib_dict[id2]
K1, R1, T1 = calib_1.K, calib_1.R, calib_1.T
K2, R2, T2 = calib_2.K, calib_2.R, calib_2.T

img1_path = f'{src}/{scene}/images/{id1}.jpg'
img2_path = f'{src}/{scene}/images/{id2}.jpg'

img1 = load_image(img1_path)
img2 = load_image(img2_path)

flag_manually = True
if flag_manually:
    # Manually clicked matching point is id1 ad id2
    points1 = np.array([[341,242],[920,247],
                        [144,283],[867,289]])

    points2 = np.array([[344,256],[968,211],
                        [201,306],[931,262]])

else:
    num_features = 5000
    sift_detector = cv2.SIFT_create(num_features, contrastThreshold=-10000, edgeThreshold=-10000)

    keypoints_1, descriptors_1 = utils.ExtractSiftFeatures(img1, sift_detector, 2000)
    keypoints_2, descriptors_2 = utils.ExtractSiftFeatures(img2, sift_detector, 2000)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    cv_matches = bf.match(descriptors_1, descriptors_2)
    cv_matches = sorted(cv_matches, key=lambda x: x.distance)
    matches = np.array([[m.queryIdx, m.trainIdx] for m in cv_matches])

    # Convert keypoints and matches to something more human-readable.
    cur_kp_1 = utils.ArrayFromCvKps(keypoints_1)
    cur_kp_2 = utils.ArrayFromCvKps(keypoints_2)
    # take top 10 best matches
    points1= cur_kp_1[matches[:, 0]][:10]
    points2 = cur_kp_2[matches[:, 1]][:10]

# points triangulation, 3d in the world and camera-1
pts, pts1 = utils.triangulate_points(K1, K2, R1, R2, T1, T2,
                                points1, points2)

# check if inferred depth is correct, when used to project from the world to camera:
points1_proj = utils.reproject_points( K1, R1, T1, pts)

print('cam1 original:' , points1)
print('cam1 projected:', points1_proj)
print('-----------')
print('3d points, cam1:', pts1)
print('-----------')

# check if inferred depth is correct, when used to project from cam1 to cam2:
depth = pts1[-1].reshape(1,-1)
points2_proj = utils.project_points(points1, depth,
                          K1, K2, R1, R2,
                          T1.reshape(3,1), T2.reshape(3,1))

print('cam2 original:' , points2)
print('cam2 projected:', points2_proj)


# Prepare to make depth inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- cnn depth for the image id1: midas or dpt
midas_flag = False
if midas_flag:
    # Model definition
    model_type = "DPT_Large"  # or "MiDaS_small" for a lighter version
    midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    midas.eval()
    midas.to(device)

    img1, inv_depth = utils.get_midas_depth(img1_path, midas, show_flag=True)
    depth_cnn_image = 1.0/(inv_depth + utils.eps)
    depth_cnn_image = depth_cnn_image.numpy()
else:
    img1, depth_cnn_image = get_dpt_depth(img1_path)
    depth_cnn_image = 1.0/(depth_cnn_image + utils.eps)

print('cnn_image_shape', depth_cnn_image.shape)
print('image', img1.shape)

# Find scale and shift to match between cnn_depth and real metric one
N, dim = points1.shape
depth_cnn = np.zeros((1, N))

for j in range(N):
    pt = points1[j]
    depth_cnn[0,j] = depth_cnn_image[math.floor(pt[1]), math.floor(pt[0])]

scale, shift = np.polyfit(depth_cnn.flatten(), depth.flatten(), 1)

depth_cnn_metric =  scale*depth_cnn + shift
print('------------')
print('real_depth', depth)
print('cnn_to_depth', depth_cnn_metric)

# Show warped image1 to reproject to camera2:
warped_image = utils.warp_image(img1, img2, scale*depth_cnn_image + shift,
    K1, K2, R1, R2,
    T1.reshape(3,1), T2.reshape(3,1))

depth_title = 'midas' if midas_flag else 'dpt'
selection_title = 'manual' if  flag_manually else 'sift'
case_title =f'{depth_title}:{selection_title}: '

show_images([img1, img2, warped_image],
            [case_title+'im1', 'im2', 'im2->1'],
            nsize=[1,3])