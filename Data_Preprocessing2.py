import cv2
import numpy as np
import os


def read_dataset(data_type):
    img_count = 0
    data_set = []
    labels = []
    root = os.path.join('./data/', data_type)
    max_pixels = 0

    for participant in os.listdir(root):
        participant_dir = os.path.join(root, participant)
        if os.path.isfile(participant_dir):
            continue
        for symbol in os.listdir(participant_dir):
            symbol_dir = os.path.join(participant_dir, symbol)
            if os.path.isfile(symbol_dir):
                continue
            for img in os.listdir(symbol_dir):
                img_path = os.path.join(symbol_dir, img)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = gray[3:gray.shape[0], 0:gray.shape[1]]
                blurred = 255-cv2.GaussianBlur(gray, (5, 5), 0)
                dst = cv2.Canny(blurred, 0, 150)
                img_thresh = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,
                                                   2)
                Contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cnts = np.concatenate(Contours)
                X, Y, W, H = cv2.boundingRect(cnts)
                cropped_image = blurred[Y:Y + H, X:X + W]
                dim_cropped = cropped_image.shape
                for i in range(1):
                    if dim_cropped[i] > max_pixels:
                        max_pixels = dim_cropped[i]
                img_count = img_count + 1
                lbl_out = np.zeros(10)
                lbl_out[int(symbol)] = 1
                print(img_count)
                data_set.append(cropped_image/np.max(cropped_image))
                labels.append(lbl_out)
    labels = np.array(labels)
    return data_set, labels, max_pixels


def reshape_images(dataset, dim):
    new_data = [];
    for i in range(len(dataset)):
        new_data.append(cv2.resize(dataset[i], dsize=(dim, dim), interpolation=cv2.INTER_CUBIC))
    dataset = np.array(new_data)
    data = np.array(dataset)  # np array ashan feeh features ktiir
    np.save('data_set', data)
    return data

def padd_images(dataset,dim):
    new_data = []
    padding_size= dim

    for i in range(len(dataset)):
        dim_cropped = dataset[i].shape
        x_padding = (padding_size - dim_cropped[1])
        y_padding = (padding_size - dim_cropped[0])
        if (x_padding % 2) == 0:
            left = int(x_padding / 2)
            right = left
        else:
            left = int(x_padding / 2 - 0.5)
            right = left + 1
        if (y_padding % 2) == 0:
            top = int(y_padding / 2)
            bottom = top
        else:
            top = int(y_padding / 2)
            bottom = top + 1

        padded_image = cv2.copyMakeBorder(dataset[i], top=top, bottom=bottom, left=left, right=right,
                                          borderType=cv2.BORDER_CONSTANT, value=0)
        new_data.append(padded_image)

    data = np.array(new_data)  # np array ashan feeh features ktiir
    #np.save('data_set', data)
    return data


def flatten(data):
    return np.reshape(data, [-1, data.shape[1] * data.shape[2]])  # 1d array
