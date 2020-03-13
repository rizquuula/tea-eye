from keras.models import load_model
from PreprocessingImage import preprocessing
import cv2
import numpy as np
import os

def TestingModel(modelName):
    # model = load_model('TeaEyeRazif-1000epochs.h5')
    model = load_model(modelName)
    # model.load_weights('BismillahFirst-5epochs.h5')

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    # img = cv2.imread('TeaEyeResized.jpg')
    # img = cv2.imread('ImageResized_Tertutup_Kiri_160.jpg')
    # img = cv2.resize(img,(20,20))
    # img = np.reshape(img,[1,20,20,1])
    imgPath = "Testing-process/0-Tertutup/ImageBGR_Testing_Tertutup190.jpg"
    TestTertutupPath = "Testing-process/0-Tertutup"
    TestTerbukaPath = "Testing-process/1-Terbuka"
    loadImgTest0 = os.listdir(TestTertutupPath)
    loadImgTest1 = os.listdir(TestTerbukaPath)
    # img = cv2.imread(imgPath)
    # cv2.imshow("Here",img)
    # cv2.waitKey(0)
    error = 0
    for img in loadImgTest0:
        imgPath = str(TestTertutupPath)+"/"+str(img)
        img = preprocessing(imgPath)
        testImg = np.reshape(img,[1,20,20,1])
        classes = model.predict_classes(testImg)

        # print("Harusnya 0 : ", classes[0][0])
        if classes[0][0] == 1:
            error+=1

    for img in loadImgTest1:
        imgPath = str(TestTerbukaPath)+"/"+str(img)
        img = preprocessing(imgPath)
        testImg = np.reshape(img,[1,20,20,1])
        classes = model.predict_classes(testImg)

        # print("Harusnya 1 : ", classes[0][0])
        if classes[0][0] == 0:
            error+=1
    print("With model name = ",modelName)
    print("Error = ",error)
    print("Percentage Err = ",error/100," %")
    print("--------------------")

listModel = ['TeaEyeRazif-1epochs.h5',
        'TeaEyeRazif-5epochs.h5',
        'TeaEyeRazif-10epochs.h5',
        'TeaEyeRazif-15epochs.h5',
        'TeaEyeRazif-20epochs.h5',
        'TeaEyeRazif-50epochs.h5',
        'TeaEyeRazif-100epochs.h5'
        ]
for m in listModel:
    TestingModel(m)