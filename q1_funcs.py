import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial import Delaunay

def read_points(file):
    f = open(file)
    n = int(f.readline())
    points = []
    for line in f.readlines():
        point = line.split(" ")
        [x, y] = int(point[0]), int(point[1])
        points.append([x, y])
    return n, points

def generate_additional_points(img):
    h, w = img.shape[0], img.shape[1]
    h2, w2 = h // 2, w // 2
    points = [[0, 0], [0, h2], [0, h - 1], [w2, h - 1], [w - 1, h - 1], [w - 1, h2], [w - 1, 0], [w2, 0]]
    return points

''' generating triangles '''

def delaunay_triangulation(points):
    triangles = Delaunay(points)
    return triangles


''' warping triangles '''

def find_morphed_triangle_points(points1, points2, alpha):
    result = []
    for i in range(points1.shape[0]):
        x = int((1 - alpha) * points1[i, 0] + alpha * points2[i, 0])
        y = int((1 - alpha) * points1[i, 1] + alpha * points2[i, 1])
        result.append([x, y])
    return np.asarray(result)


def morph_triangles(img, tri_src, tri_dst):
    mask = np.zeros(img.shape, dtype='uint8')
    M = cv2.getAffineTransform(tri_src, tri_dst)
    dest = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), None, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
    cv2.fillConvexPoly(mask, tri_dst.astype('int'), (1.0, 1.0, 1.0), 16, 0)
    result = dest
    return result, mask


def generate_morphed_image(img1, img2, points1, points2, triangles, alpha):
    mean_points = find_morphed_triangle_points(points1, points2, alpha)
    tri1 = triangles.simplices
    result = np.zeros(img1.shape, dtype='uint8')
    for tri in tri1:
        tri_points1 = np.asarray([points1[tri[0]], points1[tri[1]], points1[tri[2]]], dtype='float32')
        tri_points2 = np.asarray([points2[tri[0]], points2[tri[1]], points2[tri[2]]], dtype='float32')
        tri_points_morph = np.asarray([mean_points[tri[0]], mean_points[tri[1]], mean_points[tri[2]]], dtype='float32')

        result1, mask = morph_triangles(img1, tri_points1, tri_points_morph)
        result2, mask = morph_triangles(img2, tri_points2, tri_points_morph)
        result = (result * (1 - mask) + (alpha * result2 + (1 - alpha) * result1) * mask).astype('uint8')

    return result


def generate_morphing_list(img1, img2, points1, points2, num_of_frames):
    result = []
    triangles = delaunay_triangulation(points1)
    delta, alpha = 1.0 / num_of_frames, 0.0
    while alpha < 1:
        img_morph = generate_morphed_image(img1, img2, points1, points2, triangles, alpha)
        result.append(img_morph)
        alpha += delta
    return result


def generate_video(results, output_path, size, fps):
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for img in results:
        video.write(img)
    video.release()
