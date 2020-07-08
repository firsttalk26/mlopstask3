from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras import metrics
import random


def convolution_layer():
    return (Convolution2D(filters=random.randint(30,40),
                          kernel_size=random.choice(((3,3),(4,4),(5,5))),
                          activation='relu',
                          input_shape=(128, 128, 3)))


def pooling_layer():
    return (MaxPooling2D(pool_size=(2, 2)))


def dense_layer1():
    return (Dense(units=random.randint(100,200), activation='relu'))

def dense_layer2():
    return (Dense(units=random.randint(50,100), activation='relu'))

def dense_layer3():
    return (Dense(units=random.randint(20,40), activation='relu'))

def dense_layer4():
    return (Dense(units=1, activation='relu'))



model=Sequential()




X = random.randint(1,5)
if X==1:
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(Flatten())
    model.add(dense_layer1())
elif X==2:
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(Flatten())
    model.add(dense_layer1())
    model.add(dense_layer2())
elif X==3:
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(Flatten())
    model.add(dense_layer1())
    model.add(dense_layer2())
    model.add(dense_layer3())
elif X==4:
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(Flatten())
    model.add(dense_layer1())
    model.add(dense_layer2())
    model.add(dense_layer3())
else:
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(convolution_layer())
    model.add(pooling_layer())
    model.add(Flatten())
    model.add(dense_layer1())
    model.add(dense_layer2())
    model.add(dense_layer3())
model.add(dense_layer4())


# In[7]:


print(model.summary())


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras_preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'cnn_dataset/training_set/',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'cnn_dataset/test_set/',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')
model_history=model.fit(
        training_set,
        steps_per_epoch=800,
        epochs=random.randint(10,20),
        validation_data=test_set,
        validation_steps=80)





print(max(model_history.history['val_accuracy']))
if (max(model_history.history['val_accuracy'])) > 0.80 :
    model.save('model.h5')
    import smtplib
    s = smtplib.SMTP('smtp.gmail.com', 587) 
    s.starttls() 
    s.login("dhirajsharma884417@gmail.com","Dhiraj@3103") 
    message = "model is created with accuracy "+str(max(model_history.history['val_accuracy']))
    s.sendmail("dhirajsharma884417@gmail.com", "firsttalk26@gmail.com", message) 
    s.quit()



    
    
    
accuracy_file = open('accuracy.txt','w+')
accuracy_file.write (str(model_history.history['accuracy']))
accuracy_file.close()
                   
