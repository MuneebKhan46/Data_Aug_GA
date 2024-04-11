import cv2
import sys
import os
import math
import numpy as np
import scipy

def load_yuv420_image(filename, width, height):
    
    with open(filename, 'rb') as f:
        yuv_data = f.read()
    
    
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    
    
    y_plane = np.frombuffer(yuv_data[:y_size], dtype=np.uint8).reshape((height, width))
#     u_plane = np.frombuffer(yuv_data[y_size:y_size+uv_size], dtype=np.uint8).reshape((height // 2, width // 2))
#     v_plane = np.frombuffer(yuv_data[y_size+uv_size:], dtype=np.uint8).reshape((height // 2, width // 2))
    
    return y_plane


def ghosting_metric(fused_image, edge_image, d):
    laplacian = cv2.Laplacian(edge_image, cv2.CV_64F)
    cv2.imshow('laplacian', laplacian)

    height = edge_image.shape[0]
    width = edge_image.shape[1]

    mask_height = height // d
    mask_width = width // d

    result_image = np.zeros((mask_height, mask_width), dtype = np.uint8) 

    for j in range(mask_height):
        for i in range(mask_width):
            if j + d >= height and i + d < width:
                patch_ij = edge_image[j*d:-1, i*d:i*d+d]
                fused_ij = fused_image[j*d:-1, i*d:i*d+d]
            elif j + d >= height and i + d >= width:
                patch_ij = edge_image[j*d:-1, i*d:-1]
                fused_ij = fused_image[j*d:-1, i*d:-1]
            elif j + d < height and i + d >= width:
                patch_ij = edge_image[j*d:j*d+d, i*d:-1]
                fused_ij = fused_image[j*d:j*d+d, i*d:-1]
            else:
                patch_ij = edge_image[j*d:j*d+d, i*d:i*d+d]
                fused_ij = fused_image[j*d:j*d+d, i*d:i*d+d]

            count = cv2.countNonZero(patch_ij)
            if count >= d:
                cv2.imshow('patch_ij', patch_ij)
                n, labels = cv2.connectedComponents(patch_ij)
                if n < 4:
                    result_image[j][i] = 0
                else:
                    #print(labels)
                    avg_color = [[0, 0, 0] for _ in range(n)]
                    freq = [0] * n
                    patch_h = patch_ij.shape[0]
                    patch_w = patch_ij.shape[1]
                    for l in range(1, n):
                        for y in range(patch_h):
                            for x in range(patch_w):
                                if labels[y][x] == l:
                                    avg_color[l] = avg_color[l] + fused_ij[y][x]
                                    freq[l] = freq[l] + 1
                    for k in range(1, n):
                        avg_color[k] = avg_color[k] / freq[k]
                        #print('group {} = {}'.format(k, avg_color[k]))

                    # for each combination of lines
                    found = False
                    for a in range(1, n - 2):
                        if found:
                            break
                        for b in range(a + 1, n - 1):
                            if found:
                                break
                            for c in range(b + 1, n):
                                if found:
                                    break

                                lambda_ = []
                                mab = np.array([[avg_color[a][0], avg_color[b][0]], [avg_color[a][1], avg_color[b][1]]])
                                mc = np.array([avg_color[c][0], avg_color[c][1]])
                                if (mab[0][0] != 0 or mab[1][0] != 0) and (mab[0][1] != 0 or mab[1][1] != 0) and (mab[1][0] != 0 and mab[1][1] != 0) and (mab[0][0] / mab[1][0] != mab[0][1] / mab[1][1]):
                                    lambda_.append(scipy.linalg.solve(mab, mc))

                                mab = np.array([[avg_color[a][0], avg_color[b][0]], [avg_color[a][2], avg_color[b][2]]])
                                mc = np.array([avg_color[c][0], avg_color[c][2]])
                                if (mab[0][0] != 0 or mab[1][0] != 0) and (mab[0][1] != 0 or mab[1][1] != 0) and (mab[1][0] != 0 and mab[1][1] != 0) and (mab[0][0] / mab[1][0] != mab[0][1] / mab[1][1]):
                                    lambda_.append(scipy.linalg.solve(mab, mc))

                                mab = np.array([[avg_color[a][1], avg_color[b][1]], [avg_color[a][2], avg_color[b][2]]])
                                mc = np.array([avg_color[c][1], avg_color[c][2]])
                                if (mab[0][0] != 0 or mab[1][0] != 0) and (mab[0][1] != 0 or mab[1][1] != 0) and (mab[1][0] != 0 and mab[1][1] != 0) and (mab[0][0] / mab[1][0] != mab[0][1] / mab[1][1]):
                                    lambda_.append(scipy.linalg.solve(mab, mc))

                                if len(lambda_) > 0:
                                    avg = np.average(lambda_, axis = 0)
                                    if math.fabs(avg[0] + avg[1] - 1) <= 0.05:
                                        found = True
                                        #print('>>> Ghost found!')
                                        result_image[j][i] = 1

    return result_image

def gen_mask(image, d):
    height = image.shape[0]
    width = image.shape[1]

    mask_height = height // d
    mask_width = width // d

    mask = np.zeros((mask_height, mask_width), dtype = np.uint8) 

    for j in range(mask_height):
        for i in range(mask_width):
            if j + d >= height and i + d < width:
                patch_ij = image[j*d:-1, i*d:i*d+d]
            elif j + d >= height and i + d >= width:
                patch_ij = image[j*d:-1, i*d:-1]
            elif j + d < height and i + d >= width:
                patch_ij = image[j*d:j*d+d, i*d:-1]
            else:
                patch_ij = image[j*d:j*d+d, i*d:i*d+d]

            count = cv2.countNonZero(patch_ij)
            if count >= d:
                mask[j][i] = 255
    return mask



source_frame_path = '/Dataset/dataset_patch_raw_ver3/original/original_amusement_park1_p64_t0.3_n0_001.raw'
fused_image_path  = '/Dataset/dataset_patch_raw_ver3/denoised/denoised_amusement_park1_p64_t0.3_n0_001.raw'

source_frame = load_yuv420_image(source_frame_path, 720, 1280)
fused_image = load_yuv420_image(fused_image_path, 720, 1280)


blurred_image = cv2.blur(fused_image, (3, 3))
canny = cv2.Canny(fused_image, 20, 40)

d = 250
mask = gen_mask(canny, d)



detected_image = ghosting_metric(fused_image, canny, d)
count = cv2.countNonZero(detected_image)

print('Number of ghosting patches = {}, {:.2f}%'.format(count, 100 * count / (720 * 12800 / (d * d))))

ghost_image = np.stack([fused_image]*3, axis=-1)  # Convert grayscale back to RGB for visualization
for y in range((1080 // d) * d):
    for x in range((1920 // d) * d):
        if detected_image[y // d][x // d] == 1:
            ghost_image[y, x] = (0, 0, 255)



save_directory = '/Dataset/Berger_metric'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)


detected_file = os.path.join(save_directory, '{}_detected.png'.format(os.path.basename(fused_image_path).split('.')[0]))
cv2.imwrite(detected_file, ghost_image)


