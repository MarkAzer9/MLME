import cv2
import numpy as np
import os
import cv2 as cv
from numpy import random
import matplotlib.pyplot as plt

def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img

def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def horizontal_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift)]
    if ratio < 0:
        img = img[:, int(-1*to_shift):]
    img = fill(img, h, w)
    return img

def vertical_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :]
    if ratio < 0:
        img = img[int(-1*to_shift):, :]
    img = fill(img, h, w)
    return img

def read_dataset(data_type,dim):
    img_count=0
    data_set=[]
    labels=[]
    root=os.path.join('D:/Masters/TU Dortmund/2nd Semester/Machine Learning/Project 3 materials/project_files/data/', data_type)


    for participant in os.listdir(root):
        participant_dir = os.path.join(root, participant)
        if os.path.isfile(participant_dir):
            continue
        for symbol in os.listdir(participant_dir):
            symbol_dir=os.path.join(participant_dir,symbol)
            if os.path.isfile(symbol_dir):
                continue
            for img in os.listdir(symbol_dir):
                img_path=os.path.join(symbol_dir, img)
                img=cv2.imread(img_path)
                gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kernel_aya = np.ones((5, 5), np.float32) / 25
                imgray = (255-(cv2.filter2D(gray, -1, kernel_aya))) # avg filter
                ######################################
                canny_output = cv.Canny(imgray, 20, 200)
                contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                height = canny_output.shape[0]
                width = canny_output.shape[1]
                min_x, min_y = width, height
                max_x = max_y = 0

                for cnt in contours:
                    x, y, w, h = cv.boundingRect(cnt)
                    min_x, max_x = min(x, min_x), max(x + w, max_x)
                    min_y, max_y = min(y, min_y), max(y + h, max_y)

                if max_x - min_x > 0 and max_y - min_y > 0:
                    if (max_x - min_x) > (max_y - min_y):
                        mid_y = min_y + (max_y - min_y) / 2
                        dst = imgray[int(mid_y - ((max_x - min_x) / 2)) - 5 if int(mid_y - ((max_x - min_x) / 2)) - 5 >0 else 0: int(mid_y + ((max_x - min_x) / 2)) + 5,
                                  min_x - 5 if min_x - 5 > 0 else 0 : max_x+5]
                    else:
                        mid_x = min_x + ((max_x - min_x) / 2)
                        dst = imgray[min_y -5 if min_y -5 >0 else 0 : max_y + 5,
                                  int(mid_x - ((max_y - min_y) / 2)) - 5 if int(mid_x - ((max_y - min_y) / 2)) -5 > 0 else 0  :int(mid_x + ((max_y - min_y) / 2)) + 5]
                else:
                    dst= (255-(cv2.filter2D(gray, -1, kernel_aya)))
                ########################################################
                (thresh, blackAndWhiteImage) = cv2.threshold(dst, 20, 255, cv2.THRESH_BINARY)
                Im_padded=cv2.copyMakeBorder(blackAndWhiteImage, 5, 5, 5, 5, cv2.BORDER_CONSTANT, None, value = 0)

                res=cv2.resize(Im_padded, dsize=(dim, dim), interpolation=cv2.INTER_CUBIC)

                img_count=img_count+1
                lbl_out = np.zeros(10)
                lbl_out[int(symbol)] = 1
                print(img_count)
                data_set.append(res/np.max(res))
                #labels.append(lbl_out) #hot encoded for NN
                labels.append(symbol) #regular for SVM/KNN








    data=np.array(data_set) #np array ashan feeh features ktiir
    labels=np.array(labels)
    np.save('data_set',data)
    np.save('labels',labels)
    return data,labels

def flatten(data):
    return np.reshape(data,[-1,data.shape[1]*data.shape[2]]) #1d array
def plot25(train_data):
    indexes = np.random.randint(0, train_data.shape[0], size=25)
    images = train_data[indexes]

    # plot 25 random digits
    plt.figure(figsize=(5, 5))
    for i in range(len(indexes)):
        plt.subplot(5, 5, i + 1)
        image = images[i]
        plt.imshow(image, cmap='gray')
        plt.axis('off')