
import numpy as np
import cv2
import csv
from glob import glob
import matplotlib.pyplot as plt
import os

from PIL import Image
import torch
import cv2
from romatch import roma_outdoor

from skimage import io, exposure


def FlattenMatrix(M, num_digits=8):
    '''Convenience function to write CSV files.'''

    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])


def replace_with_he_image(im1_path, im2_path, data_folder):
    source = io.imread(im2_path)
    reference = io.imread(im1_path)

    matched = exposure.match_histograms(source, reference, channel_axis=2)
    fname2 = os.path.basename(im2_path)
    temp_path = os.path.join(data_folder, 'temp', fname2)
    io.imsave(temp_path, matched)
    return temp_path


def crop_subimage(im_path, iw, ih, grid_index, temp_folder='temp'):
    source = io.imread(im_path)
    fname = os.path.basename(im_path)
    fname_id, format = fname.split('.')
    fname_crop = fname_id + '_' + grid_index + '.'+ format
    os.makedirs(os.path.join(data_folder, temp_folder), exist_ok=True)
    temp_path = os.path.join(data_folder, temp_folder, fname_crop)
    io.imsave(temp_path, source[ih[0]:ih[1], iw[0]:iw[1]])
    return temp_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')


# Read the pairs file.
data_folder = '/home/inna/match_code/RoMa/tourist_data'
roma_model = roma_outdoor(device=device)
#roma_model.sample_thresh = 0.05

# Read Data:
test_samples = []
csv_file = os.path.join(data_folder, 'test.csv')
with open(csv_file) as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        # Skip header.
        if i == 0:
            continue
        test_samples += [row]

# how_many_to_fill = 500
how_many_to_fill = -1
F_dict = {}

for i, row in enumerate(test_samples):

    if i % 500 == 0:
        print(i)

    sample_id, batch_id, image_1_id, image_2_id = row

    # Load the images.

    im1_path = f'{data_folder}/test_images/{batch_id}/{image_1_id}.jpg'
    im2_path = f'{data_folder}/test_images/{batch_id}/{image_2_id}.jpg'

    W_A, H_A = Image.open(im1_path).size
    W_B, H_B = Image.open(im2_path).size

    # Extract dense features and match:
    warp, certainty = roma_model.match(im1_path, im2_path, device=device)
    # Sample matches for estimation
    matches, certainty = roma_model.sample(warp, certainty)
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

    # ---- Split the first image:
    half_w_a, half_h_a = W_A//2, H_A//2
    pad_w, pad_h =  W_A//4, H_A//4

    idx_x = {0: [0, half_w_a + pad_w], 1: [half_w_a - pad_w, W_A-1]}
    idx_y = {0: [0, half_h_a + pad_h], 1: [half_h_a - pad_h, H_A-1]}

    for k in [0, 1]:
        for j in [0, 1]:
            im_temp_path = crop_subimage(im1_path, idx_x[k], idx_y[j],grid_index='{}{}'.format(k, j))
            W, H = Image.open(im_temp_path).size
            # Extract dense features and match:
            warp, certainty = roma_model.match(im_temp_path, im2_path, device=device)
            # Sample matches for estimation
            matches, certainty_ = roma_model.sample(warp, certainty)
            kpts1_, kpts2_ = roma_model.to_pixel_coordinates(matches, H, W, H_B, W_B)
            kpts1_[:, 0] = + idx_x[k][0]
            kpts1_[:, 1] = + idx_y[j][0]

            kpts1 = torch.vstack([kpts1, kpts1_])
            kpts2 = torch.vstack([kpts2, kpts2_])

    """
    # From my code:
    F, mask = cv2.findFundamentalMat(
        kpts1.cpu().numpy(), kpts2.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC,
        confidence=0.999999, maxIters=10000
    )
    """
    try:
        F, inlier_mask = cv2.findFundamentalMat(
            kpts1.cpu().numpy(), kpts2.cpu().numpy(),
            method=cv2.USAC_MAGSAC, ransacReprojThreshold=0.25,
            confidence=0.99999, maxIters=100000)
    except:
        F = np.random.rand(3, 3)
        print('Something went wrong: ', sample_id, im1_path, im2_path)

    F_dict[sample_id] = F
    # os.remove(im2_he_path)

with open('submission_allpoints.csv', 'w') as f:
    f.write('sample_id,fundamental_matrix\n')
    for sample_id, F in F_dict.items():
        f.write(f'{sample_id},{FlattenMatrix(F)}\n')
