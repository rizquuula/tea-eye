import cv2
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import time 
import os
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

GENERATE_NAME = "CNN-TeaEye-{}".format(int(time.time()))
callbacks = TensorBoard(log_dir='./Graph/{}'.format(GENERATE_NAME))

dirTrain = 'Training-Process/Training-Database/'
dirValidation = 'Training-Process/Validating-Database/'
dirTrainTertutup = 'Training-Process/Training-Database/0-Tertutup/'
dirTrainTerbuka = 'Training-Process/Training-Database/1-Terbuka/'
dirValidationTertutup = 'Training-Process/Validating-Database/0-Tertutup/'
dirValidationTerbuka = 'Training-Process/Validating-Database/1-Terbuka/'

training = 300*2
validating = 100*2
batch_size = 32
img_size = 20
epoch = 100

def preprocessing(img_source):
    img = cv2.imread(img_source)
    return new_img

def preprocessing_old(img_source):
    #img_source = str(img_source)
    img = cv2.imread(img_source)

    old_size = img.shape[:2]       #Original size
    #print(old_size)     
    # => (288, 352)
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])      #Changed to the new size in same ratio
    #print(ratio,' and ',new_size)      
    # => 0.6363636363636364  and  (183, 224)#
    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_h = img_size - new_size[0]
    delta_w = img_size - new_size[1]
    #print(delta_w,' and ',delta_h)
    # => 0 and 41
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    #print (top,bottom,left,right)
    # => 20 21 0 0                  // is for integer divide

    #color = [255, 255, 255]
    color = [0, 0, 0]

    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    #print(new_img.shape)
    #print(new_img)
    #new_img.reshape(-1,img_size,img_size,1)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    #print(new_img.shape)
    '''a = np.array([[1,2,3], [4,5,6], [7,8,9]])
    print(a)
    a.reshape(-1,2,2,2)
    print(a)'''
    '''
    cv2.imshow('This is image',new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    return new_img


'''cv2.imshow('This is image',preprocessing('Resizeme.jpg'))
cv2.waitKey(0)
cv2.destroyAllWindows()'''

def layerprocess_NoKeras(size,dimension):
    convnet = input_data(shape=[None, size, size, dimension], name='input')
    
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')
    return model

def layerprocess(size,dimension):
    inputs = Input(shape=(28, 28, 1))
    conv_layer = ZeroPadding2D(padding=(2,2))(inputs) 
    conv_layer = Conv2D(16, (3, 3), strides=(3,3), activation='relu')(conv_layer) 
    conv_layer = MaxPooling2D((2, 2))(conv_layer) 
    conv_layer = Conv2D(32, (3, 3), strides=(1,1), activation='relu')(conv_layer) 
    # conv_layer = ZeroPadding2D(padding=(1,1))(conv_layer) 
    # conv_layer = Conv2D(64, (3, 3), strides=(1,1), activation='relu')(conv_layer)

    # Flatten feature map to Vector with 576 element.
    flatten = Flatten()(conv_layer) 

    # Fully Connected Layer
    fc_layer = Dense(256, activation='relu')(flatten)
    fc_layer = Dense(32, activation='relu')(fc_layer)
    outputs = Dense(1, activation='softmax')(fc_layer)

    model = Model(inputs=inputs, outputs=outputs)

    return model
#Import image
load_Tertutup_img_train = os.listdir(dirTrainTertutup)
load_Terbuka_img_train = os.listdir(dirTrainTerbuka)

load_Tertutup_img_val = os.listdir(dirValidationTertutup)
load_Terbuka_img_val = os.listdir(dirValidationTerbuka)
'''
for img in load_Z_img_train:
    #print(str(load_Z_img_train)+str(img))
    print(str(dirTrainTertutup)+str(img))
'''
'''
for img in tqdm(load_Z_img_train):
    #print(str(load_Z_img_train+img))
    img_array = preprocessing(str(dirTrainTertutup)+str(load_Z_img_train)+str(img))
    cv2.imshow('This is image',img_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''

def create_train_data():
    training_data = []
    for img in tqdm(load_Tertutup_img_train):
        #print(str(load_Z_img_train+img))
        img_array = preprocessing(str(dirTrainTertutup)+str(img))
        label = [1,0]
        training_data.append([np.array(img_array) , np.array(label)])

    for img in tqdm(load_Terbuka_img_train):
        img_array = preprocessing(str(dirTrainTerbuka)+str(img))
        label = [0,1]
        training_data.append([np.array(img_array) , np.array(label)])

    np.save('training_data.npy',training_data)
    return training_data

def create_val_data():
    validation_data = []
    for img in tqdm(load_Tertutup_img_val):
        #print(str(load_Z_img_val+img))
        img_array = preprocessing(str(dirValidationTertutup)+str(img))
        label = [1,0]
        validation_data.append([np.array(img_array) , np.array(label)])

    for img in tqdm(load_Terbuka_img_val):
        img_array = preprocessing(str(dirValidationTerbuka)+str(img))
        label = [0,1]
        validation_data.append([np.array(img_array) , np.array(label)])
    np.save('val_data.npy', validation_data)
    return validation_data

#train = create_train_data()
#val = create_val_data()

#If you already have database
train = np.load('/home/linkgish/Desktop/MachLearn5/training_data.npy')
val = np.load('/home/linkgish/Desktop/MachLearn5/val_data.npy')

X = np.array([i[0] for i in train]).reshape(-1,img_size,img_size,1)
Y = [i[1] for i in train]

val_x = np.array([i[0] for i in val]).reshape(-1,img_size,img_size,1)
val_y = [i[1] for i in val]

model = layerprocess(img_size,1)

adam = Adam(lr=0.0001)
#model.compile(
#    optimizer=adam, 
#    loss='categorical_crossentropy', 
#    metrics=['accuracy'])

model.fit({'input': X}, {'targets': Y}, n_epoch=epoch, 
    validation_set=({'input': val_x}, {'targets': val_y}), 
    snapshot_step=500, show_metric=True, run_id=GENERATE_NAME)
#model.fit({'input': X}, {'targets': Y}, 
#    n_epoch=3, 
#    validation_set=({'input': test_x}, {'targets': test_y}), 
#    snapshot_step=500, 
#    show_metric=True, 
#    callbacks=[callbacks])

#model.save_weights('SignZeroFive.h5')
model.save('SignZeroFive2.model')
#model.make_model("SignZeroFive.json")