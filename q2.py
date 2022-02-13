import numpy as np
import cv2
import q2_funcs as funcs

img1 = cv2.imread("images/res05.jpg")
img2 = cv2.imread("images/res06.jpg")

img1 = (img1 / 255).astype('float32')
img2 = (img2 / 255).astype('float32')

source = img1[195:485, 355:715, :].astype('float32')
target = img2[195:485, 355:715, :].astype('float32')

mask = np.zeros((290, 360), dtype='float32')
mask[5:-5, 5:-5] = 1

max_matrix, min_matrix = np.ones(source.shape, dtype='float32') * 255, np.zeros(source.shape, dtype='float32')
result = funcs.blend_images(source, target, mask) * 255
result = np.minimum(result, max_matrix)
result = np.maximum(result, min_matrix)
result = result.astype('uint8')
final_result = (img2.copy() * 255).astype('uint8')
final_result[195:485, 355:715, :] = result
cv2.imwrite("images/res07.jpg", final_result)
