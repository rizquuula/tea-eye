from keras.models import load_model
import cv2
import numpy as np

model = load_model('BismillahFirst-5epochs.h5')
# model.load_weights('BismillahFirst-5epochs.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# img = cv2.imread('TeaEyeResized.jpg')
img = cv2.imread('ImageResized_Tertutup_Kiri_160.jpg')

img = cv2.resize(img,(20,20))
img = np.reshape(img,[1,20,20,3])

classes = model.predict_classes(img)

print(classes)