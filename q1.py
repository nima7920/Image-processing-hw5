import numpy as np
import cv2
import q1_funcs as funcs

img1 = cv2.imread("images/res01.jpg")
img2 = cv2.imread("images/res02.jpg")

''' reading points'''
n, points1 = funcs.read_points("images/tom.txt")
n, points2 = funcs.read_points("images/leo.txt")

''' extending points '''
points1.extend(funcs.generate_additional_points(img1))
points2.extend(funcs.generate_additional_points(img2))

points1 = np.asarray(points1, dtype='int')
points2 = np.asarray(points2, dtype='int')

''' creating final result '''
result = []
images = funcs.generate_morphing_list(img1, img2, points1, points2, 45)
cv2.imwrite("images/res03.jpg", images[15])
cv2.imwrite("images/res04.jpg", images[30])
result.extend(images)
images.reverse()
result.extend(images)
funcs.generate_video(result, "images/morph.mp4", (img1.shape[1], img2.shape[0]), 30)
