
import numpy as np
from PIL import Image
import cv2

def add_gaussian_noise(image, sigma):
    
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    return noisy_image

def add_poission_noise(image, sigma):
    noise = np.random.poisson(sigma, image.shape)
    noisy_image = image + noise
    return noisy_image

if __name__=='__main__':
    image_path = 'img/test.jpeg'
    image = cv2.imread(image_path, 0)
    cv2.imwrite('img/original_image.jpg', image)

    noisy_image = add_gaussian_noise(image, 5)
    cv2.imwrite('img/noisy_image_gaussian.jpg', noisy_image)

    noisy_image = add_poission_noise(image, 5)
    cv2.imwrite('img/noisy_image_poission.jpg', noisy_image)

