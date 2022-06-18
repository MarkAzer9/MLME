import cv2
import numpy as np
import os

def read_dataset3(data_type,dim):
    img_count=0
    data_set=[]
    labels=[]
    root=os.path.join('./data/', data_type)
    padding_size = 96


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
                #kernel_aya = np.ones((5, 5), np.float32) / 25
                #dst = (255-(cv2.filter2D(gray, -1, kernel_aya)))
                dst = (255 - (cv2.bilateralFilter(gray, 9, 75, 75)))
                MIN_CONTOUR_AREA = 200
                img_thresh = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,
                                                   2)
                Contours, imgContours = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for contour in Contours:
                    if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                        [X, Y, W, H] = cv2.boundingRect(contour)

                cropped_image = dst[Y:Y + H, X:X + W]

                dim_cropped = cropped_image.shape
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

                padded_image = cv2.copyMakeBorder(cropped_image, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_CONSTANT, value=0)

                img_count=img_count+1
                lbl_out = np.zeros(10)
                lbl_out[int(symbol)] = 1
                print(img_count)
                data_set.append(padded_image/np.max(padded_image))
                labels.append(lbl_out)

    data=np.array(data_set) #np array ashan feeh features ktiir
    labels=np.array(labels)
    np.save('data_set',data)
    np.save('labels',labels)

    return data, labels

def flatten(data):
    return np.reshape(data,[-1,data.shape[1]*data.shape[2]]) #1d array
