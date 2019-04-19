import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from src.utility import load_cuave


def crop_mouth_region_cv2(verbose=True):
    import cv2
    # load pre-trained cascade classifier
    mouth_cascade = cv2.CascadeClassifier('./models/haarcascade_smile.xml')

    _, _, _, frames_1, frames_2, _ = load_cuave()
    count_1, count_2 = 0, 0

    vis = 1
    for i in range(len(frames_1)):
        for j in range(len(frames_1[i])):
            img = frames_1[i][j]
            img = cv2.resize(img, (150, 100))
            mouth = None
            mouth_rects = mouth_cascade.detectMultiScale(img, 1.1, 11)
            for (ex,ey,ew,eh) in mouth_rects:
                roi = img[ey:ey+eh, ex:ex+ew]
                mouth = cv2.resize(roi, (60,20))
                if vis == 5000 and verbose:
                    io.imshow(mouth)
                    plt.show()
                    vis = 1
                else:
                    vis += 1
                count_1 += 1
                break
    
    assert count_1 == len(frames_1) * len(frames_1[0])

    vis = 1
    for i in range(len(frames_2)):
        for j in range(len(frames_2[i])):
            img = frames_2[i][j]
            img = cv2.resize(img, (150, 100))
            mouth = None
            mouth_rects = mouth_cascade.detectMultiScale(img, 1.1, 11)
            for (ex,ey,ew,eh) in mouth_rects:
                roi = img[ey:ey+eh, ex:ex+ew]
                mouth = cv2.resize(roi, (60,20))
                if vis == 5000 and verbose:
                    io.imshow(mouth)
                    plt.show()
                    vis = 1
                else:
                    vis += 1
                count_1 += 1
    
    assert count_2 == len(frames_2) * len(frames_2[0])
