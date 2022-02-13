import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import q3_funcs as funcs

img1 = cv.imread("images/res08.jpg")
img2 = cv.imread("images/res09.jpg")

img_float1 = (img1 / 255).astype('float32')
img_float2 = (img2 / 255).astype('float32')

kernel_size, sigma, n = 151, 30, 10
img_list = funcs.generate_gaussian_stack(img1, kernel_size, sigma, n)
lap_list = funcs.generate_laplacian_stack(img1, kernel_size, sigma, n)

''' creating masks '''
h, w = img1.shape[0], img1.shape[1]
mask = np.zeros(img1.shape, dtype='float32')
mask[:, 0:250] = 1
masks = funcs.generate_masks(mask, 151, 30, n)

result = funcs.blend_images(img_float1, img_float2, masks, kernel_size, sigma, n)
result = funcs.convert_from_float32_to_uint8(result)
cv.imwrite("images/res10.jpg", result)
