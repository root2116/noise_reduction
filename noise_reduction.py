import numpy as np
from PIL import Image
import pandas as pd
import cv2
import math
import sys

def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return np.exp(- (x ** 2) / (2 * sigma ** 2))



def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):

    hl = diameter//2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            gi = gaussian(source[neighbour_x][neighbour_y] - source[x][y], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = int(round(i_filtered))


def bilateral_filter(original_image, r, sigma_i, sigma_s):

    filtered_image = np.zeros(original_image.shape)
    for i in range(original_image.shape[0]):
        for j in range(original_image.shape[1]):
            apply_bilateral_filter(original_image, filtered_image, i, j, r, sigma_i, sigma_s)
        
    

    return filtered_image

def calculate_weight(i, j, m, n, f, h, patch_size):
    
    sum = 0
    for s in range(-patch_size, patch_size+1):
        for t in range(-patch_size, patch_size+1):
            if i+s < 0 or i+s >= f.shape[0] or j+t < 0 or j+t >= f.shape[1]:
                continue
            if i+s+m < 0 or i+s+m >= f.shape[0] or j+t+n < 0 or j+t+n >= f.shape[1]:
                continue
            sum += (f[i+s, j+t] - f[i+s+m, j+t+n]) ** 2

            
    return np.exp(-max(sum - 2*h*h, 0) / (2 * h*h))
       
def apply_nlm_filter(img, filtered_image, i, j, r, h, w):
    weights = np.zeros((img.shape[0], img.shape[1]))
    
    for m in range(-r, r+1):
        for n in range(-r, r+1):
            if i+m < 0 or i+m >= img.shape[0] or j+n < 0 or j+n >= img.shape[1]:
                continue
            weights[i + m, j + n] = calculate_weight(i, j, m, n, img, h, w)

    
    # print(np.sum(weights))

    weights /= np.sum(weights)
    filtered_image[i, j] = np.sum(img * weights)
    
def non_local_means(original_image, r, h, w):
    filtered_image = np.zeros(original_image.shape)

    for i in range(original_image.shape[0]):
        print(i)
        for j in range(original_image.shape[1]):
            apply_nlm_filter(original_image, filtered_image, i, j, r, h, w)

    return filtered_image

# img: 画像
# r: ウィンドウの半径
def box_filter(img, r):
    (rows, cols) = img.shape
    padded_img = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT)
    integral_img = np.cumsum(np.cumsum(padded_img, axis=0), axis=1)
    result = np.zeros_like(padded_img)
    for x in range(rows + 2*r):
        for y in range(cols + 2*r):
            x1 = max(0, x-r)
            y1 = max(0, y-r)
            x2 = min(rows-1 + 2*r, x+r)
            y2 = min(cols-1 + 2*r, y+r)
            result[x,y] = integral_img[x2, y2] - integral_img[x1, y2] - integral_img[x2, y1] + integral_img[x1, y1]
    result /= ((2*r+1)**2)
    return result[r:-r, r:-r]

# I: ガイド画像
# p: 入力画像
# r: ウィンドウの半径
# eps: 正則化パラメータ
def guided_filter(I, p, r, eps):
    mean_I = box_filter(I, r)
    mean_p = box_filter(p, r)
    corr_I = box_filter(I*I, r)
    corr_Ip = box_filter(I*p, r)
    cov_Ip = corr_Ip - mean_I * mean_p
    var_I = corr_I - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)
    q = mean_a * I + mean_b
    return q

def process_image(original_image, r, eps):
    # img = np.array(Image.open(image_path)) / 255.0
    original_image = original_image / 255.0
    
    filtered_image = np.zeros(original_image.shape)
    # for i in range(3):  # RGB channels
    #     filtered_image[:, :, i] = guided_filter(img[:, :, i], img[:, :, i], r, eps)
    filtered_image = guided_filter(original_image, original_image, r, eps)
    
    filtered_image = np.clip(filtered_image * 255.0, 0, 255).astype(np.uint8)
    
    return filtered_image
    

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


if __name__=='__main__':
    # get the first argument as the image path
    image_path = sys.argv[1]
    # read the image
    original_image = cv2.imread(image_path, 0)
    cv2.imwrite(f'{sys.argv[2]}_original_image.jpg', original_image)
    # # apply the bilateral filter to the image
    # bilateral_filtered_image = bilateral_filter(original_image, 4, 25, 25)
    # # save the filtered image
    # cv2.imwrite(f'{sys.argv[2]}_bilateral_filtered_image.jpg', bilateral_filtered_image)
    
    # # apply the guided filter to the image
    # guided_filtered_image = process_image(original_image, 7, 0.05)
    # # save the filtered image
    # cv2.imwrite(f'{sys.argv[2]}_guided_filtered_image.jpg', guided_filtered_image)

    # # apply the non-local means filter to the image
    nlm_filtered_image = non_local_means(original_image, 3,550, 6)
    # save the filtered image
    cv2.imwrite(f'{sys.argv[2]}_nlm_filtered_image.jpg', nlm_filtered_image)

