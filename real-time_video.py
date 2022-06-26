import numpy as np
import cv2
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Activation,Flatten, Conv2D, MaxPooling2D
# model = tf.keras.models.load_model('F:\CODIENTU\HOCKYVI\AI\projectfinal\New folder\Foliar-diseases-in-Apple-Trees-Prediction-master\models\apple3.h5')
model = tf.keras.models.load_model(r'F:\CODIENTU\HOCKYVI\AI\projectfinal\New folder\Foliar-diseases-in-Apple-Trees-Prediction-master\models\apple3.h5')
# path = "Image_Internet/Early_blight4.jpg"
Mydict = ["Healthy1","Multiple Disease1","Rust1","Scab1"]
        

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(r'F:\CODIENTU\HOCKYVI\AI\projectfinal\New folder\Foliar-diseases-in-Apple-Trees-Prediction-master\video\scab1\scab1.mp4') 
cap.set(3, frameWidth) 
cap.set(4, frameHeight)
# img = cv2.imread(path)   #BGR

while True:
    success, img = cap.read()
    img_goc = img
    img_goc = cv2.resize(img_goc, (640,480))
    # Nhan dang anh
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img_array = np.array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    print("Predictions:",predictions)
    predicted_class = np.argmax(predictions[0])
    confidence = round(100 * (np.max(predictions[0])), 2)
    print(predicted_class)
    print(confidence)

    # Hien thi 
    cv2.putText(img_goc, f'Loai benh: {Mydict[predicted_class]}', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img_goc, f'do chinh xac: {confidence}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Result", img_goc)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

