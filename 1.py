import tensorflow
from tensorflow import keras
from keras.models import Sequential 
from keras.optimizers import RMSprop, SGD, Adam


from keras.layers import Conv2D 
from keras.layers import MaxPooling2D
from keras.layers import Flatten 

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense 
from keras.layers import BatchNormalization 
from keras.layers import Dropout
# import seaborn as sns
# from matplotlib import pyplot as plt


#BASIC CNN
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3), activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=(3,3), activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
# model.add(Conv2D(96,kernel_size=(3,3),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())
# model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

adam = Adam(lr=0.001)
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

# model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


bs=32
# train_data_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_data_gen = ImageDataGenerator(rescale=1.0/255.)
test_data_gen = ImageDataGenerator(rescale=1.0/255.)

training_set = train_data_gen.flow_from_directory(r'D:\\CNN1 - Udemy\\dataset\\dataset\\train', target_size=(128,128), batch_size=bs,class_mode='categorical')
# print(train_data_gen)
labels1 = (training_set.class_indices)
print(labels1)

testing_set = test_data_gen.flow_from_directory(r'D:\\CNN1 - Udemy\\dataset\\dataset\\test', target_size=(128,128), batch_size=bs,class_mode='categorical')
# print(test_data_gen)
labels2 = (testing_set.class_indices)
print(labels2)

history = model.fit(training_set,validation_data=testing_set,epochs=15)

#Making new predictions
model_json = model.to_json()
with open("model1.json","w") as json_file:
    json_file.write(model_json)
    #Serializing the weights
    model.save_weights("model1.h5")
    print("Saved the model to Disk")