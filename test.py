import cv2
import numpy as np
import re
import os
import pytesseract
from keras.models import model_from_json
from keras import backend
from yaml import load
from recognition import load_model, num_to_label, label_to_num
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation as inter
from skimage.transform import rotate

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)
    best_angle = angles[scores.index(max(scores))]
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

if __name__ == "__main__":
    print("\n\n Lo'p thi: 118147. \n\n")
    # img = cv2.imread("He ho tro nhap diem tu dong\Data\\ref_image.png", cv2.IMREAD_GRAYSCALE)
    # thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # cv2.imwrite("He ho tro nhap diem tu dong\Data\\ref_image.png", thresh)
    # img = cv2.imread("He ho tro nhap diem tu dong\Data\\1-1.png", cv2.IMREAD_GRAYSCALE)
    # img = correct_skew(img)
    # thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # cv2.imwrite("He ho tro nhap diem tu dong\Data\\sample.png", thresh)
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()
    # model = load_model()    

    # img = cv2.imread("He ho tro nhap diem tu dong\\Data\\test.jpg", cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (100, 40))
    # img = img / 255.0
    # pred = model.predict(img.reshape(1, 40, 100, 1))
    # decoded = backend.get_value(backend.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0])
    # diem = num_to_label(decoded[0])
    # diem = diem.replace("n", "")
    # print(diem)