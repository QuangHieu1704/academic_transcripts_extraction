import cv2
import numpy as np
import re
import os
import pandas as pd
import pytesseract
from keras.models import model_from_json
from keras import backend
from process_data import correct_skew, get_lines, image_matching
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


characters = u"0123456789.n"


# Mapping character to number
def label_to_num(label):
    num = []
    for character in label:
        num.append(characters.find(character))
    return np.array(num)


# Mapping number to character
def num_to_label(num):
    label = ""
    for ch in num:
        if ch == -1:
            break
        else:
            label += characters[ch]
    return label


def load_model():
    with open('.\\Model\\model_CRNNCTC_Res4_lr00005.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights('.\\Model\\model_CRNNCTC_Res4_lr00005.h5')
    
    return model
    

def doc_bang_diem(file_url, model):
    img = Image.open(file_url)
    img = ImageOps.grayscale(img)
    img = np.array(img)
    # All images must resize to height 1653 and width 2337
    img = cv2.resize(img, (1653, 2337), interpolation=cv2.INTER_CUBIC)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    img_filename = file_url.split("\\")[-1]
    img = correct_skew(img)
    height, width = img.shape
    horizontal_coor, vertical_coor = get_lines(img)

    # Read Class ID
    ref_img = cv2.imread("He ho tro nhap diem tu dong\\Data\\ref_image.png", cv2.IMREAD_GRAYSCALE)
    img_malop = image_matching(img, ref_img)
    img_malop = cv2.resize(img_malop, None, fx = 1.5, fy = 1.5, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(img_malop)
    malop = re.search(r'\d\d\d\d\d\d', text)
    if malop:
        malop = malop.group()
    else:
        malop = 'ERROR ' + img_filename
    excel_filename = str(malop) + '.xlsx'

    # Read student ID
    list_mssv = []
    for i in range(1, len(horizontal_coor) - 1):
        img_mssv = img[horizontal_coor[i]: horizontal_coor[i + 1], vertical_coor[1]: vertical_coor[2]]
        img_mssv = cv2.resize(img_mssv, None, fx = 1.5, fy = 1.5, interpolation=cv2.INTER_CUBIC)
        mssv = pytesseract.image_to_string(img_mssv)
        mssv = re.search(r'\d\d\d\d\d\d\d\d', mssv)
        if mssv:
            mssv = mssv.group()
        else:
            mssv = pytesseract.image_to_string(img_mssv)
        list_mssv.append(mssv)
    
    # Read score
    list_diem = []
    for i in range(1, len(horizontal_coor) - 1):
        img_diem = img[horizontal_coor[i]: horizontal_coor[i + 1] + round(height * 0.004),
                   vertical_coor[4]: vertical_coor[5]]
        img_diem = cv2.resize(img_diem, (100, 40))
        img_diem = img_diem / 255.0
        pred = model.predict(img_diem.reshape(1, 40, 100, 1))
        decoded = backend.get_value(
            backend.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0])
        diem = num_to_label(decoded[0])
        diem = diem.replace("n", "")
        list_diem.append(diem)

        pred = pred.reshape(20, 13)
        pred = np.max(pred, axis = 1)
    
    df = pd.DataFrame(columns=['MSSV', 'Điểm'])
    df['MSSV'] = pd.Series(list_mssv).values
    df['Điểm'] = pd.Series(list_diem).values

    return excel_filename, df


def recognize_file(model, path_file, output_folder):
    now = datetime.now()
    new_output = now.strftime("%d-%m-%Y %H-%M-%S")
    output_folder = os.path.join(output_folder, new_output)
    os.mkdir(output_folder)
    img_filename = path_file.split('/')[-1]
    excel_filename, df = doc_bang_diem(path_file, model)
    if os.path.exists(os.path.join(output_folder, excel_filename)) is True:
        old_df = pd.read_excel(os.path.join(output_folder, excel_filename))
        old_df = pd.DataFrame(old_df)
        new_df = pd.concat([old_df, df])
        new_df.to_excel(os.path.join(output_folder, excel_filename), index=False)
    else:
        df.to_excel(os.path.join(output_folder, excel_filename), index=False)
    
    print("Write successfully ", img_filename)


def recognize_folder(model, input_folder, output_folder):
    now = datetime.now()
    new_output = now.strftime("%d-%m-%Y %H-%M-%S")
    output_folder = os.path.join(output_folder, new_output)
    os.mkdir(output_folder)
    for img_filename in os.listdir(input_folder):
        print("read: ", img_filename)
        excel_filename, df = doc_bang_diem(os.path.join(input_folder, img_filename), model)
        if os.path.exists(os.path.join(output_folder, excel_filename)) is True:
            old_df = pd.read_excel(os.path.join(output_folder, excel_filename))
            old_df = pd.DataFrame(old_df)
            new_df = pd.concat([old_df, df])
            new_df.to_excel(os.path.join(output_folder, excel_filename), index=False)
        else:
            df.to_excel(os.path.join(output_folder, excel_filename), index=False)

        print("Write successfully ", img_filename)
