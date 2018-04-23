
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a deep learning model the clones the driving behavior 

# **In this project I have worked on a deep learning model based on LeNet architecture by Yan LeCun to clone the driving behavior of a car.**
# 
# 
# I have tested this model on Udacity simulator for a complete lap
# 
# 
# In the following I will be explaining each part of the project ... 

# ---
# ## Project Pipeline Stepts:
# 
# **Step 1:** Load The data
# 
# **Step 2:** Split data into trainig and validation sets
#     
# **Step 3:** Define a generator function to be used through training
# 
# **Step 4:** Use the defined generator in step 3 for training set and validation set
#     
# **Step 5:** Using keras, build a regression model based on LeNet architecture to predict the steering angle
#     
# In the next, I will be giving some details related to each steps

# ---
# ## Environment:
# * AWS carnd Instance
# * Python 3.6.4
# * Anaconda 4.4.10

# **Import packages**

# In[1]:


import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os


# ---
# 
# ## Step 1: Load The data
# 
# Here I am using the python csv library to read the csv file generated from the simulator.
# This file contains paths to three camera images captured through the training, steering angle, throttle, break, and vehicle speed.
# 
# We will ignore the throttle, break, and speed measurements.
# 
# We will use the images as the feature set and the steering angle as the label set.

# In[2]:


samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
print("Done")


# ---
# 
# ## Step 2: Split data into trainig and validation sets
# 
# Here I am splitting the data into two set, training set which is 80% of the data, and validation set which is 20% of the data.

# In[3]:


train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(len(train_samples))
print(len(validation_samples))
print("Done")


# ---
# 
# ## Step 3: Define a generator function to be used through training and validation
# 
# Generator is a powerful tool in python. It is used to pull pieces of the data instead of loading it at once in memory. Here in our case, we will be using generators for pre-processing data, surey pieces of them, processing them on the fly only when we need.
# 
# Here is the steps implemented inside our generator function:
#  * First of all, all the steps are encapsulated in a while(1) loop in roder to prevent the generator from termination. We need the generator alive as long as the training is alive
#  * Everytime, we shuffle the data. **Why** ? *In order not to make the model biased by the order of images*
#  * Then we get a *slice* off our data based on the batch size defined. Here we get a batch of the training samples
#  * For each batch, we get the center, left, and right camera images.
#  * Also, we get the steering angle.
#  * Here, I am usign a correction factor for the steering angles caputured from right and left cameras. The steering angle of the left should be less than the steering angle of the center. The steering angle of the right should be more that the steering angle of the center. Hence, I used this correction factor to compensate this difference.
#  * I made a data augmentation for each image and steering angle by flipping the reading.
#  * Then convert our list images and steeting angles into NumPy lists as this is the type expected by Keras.
#  * Finally, before yielding (return for geterators) the feature and label sets, we shuffle.
#  

# In[4]:


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steering_angles = []
            correction = 0.1889
            
            for batch_sample in batch_samples:
                # Reading center image and steering angle
                name = "data/IMG/"+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                if center_image is not None:
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    images.append(center_image)
                    
                    center_angle = float(batch_sample[3])
                    steering_angles.append(center_angle)
                    
                    
                    # Making data augmentation - Flippiong readings
                    center_image = np.fliplr(center_image)
                    images.append(center_image)
                    
                    center_angle = - center_angle
                    steering_angles.append(center_angle)
                    
                    
                
                else :
                    print("Center Image " +  name + " is NONE")
                    

                    
                # Reading left image and steering angle
                name = "data/IMG/"+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                if left_image is not None:
                    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                    images.append(left_image)
                    
                    center_angle = float(batch_sample[3])
                    left_angle = center_angle + correction
                    steering_angles.append(left_angle)
                    
                    
                    # Making data augmentation - Flippiong readings
                    left_image = np.fliplr(left_image)
                    images.append(left_image)
                    
                    left_angle = - left_angle
                    steering_angles.append(left_angle)
                
                else :
                    print("Left Image " +  name + " is NONE")                 
                
                
                # Reading right image and steering angle
                name = "data/IMG/"+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                if right_image is not None:
                    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                    images.append(right_image)
                    
                    center_angle = float(batch_sample[3])
                    right_angle = center_angle - correction
                    steering_angles.append(right_angle)
                    
                    
                    # Making data augmentation - Flippiong readings
                    right_image = np.fliplr(right_image)
                    images.append(right_image)
                    
                    right_angle = - right_angle
                    steering_angles.append(right_angle)
                
                else :
                    print("Right Image " +  name + " is NONE")  
                    

            X_train = np.array(images)
            y_train = np.array(steering_angles)
            
            yield shuffle(X_train, y_train)


# ---
# 
# ## Step 4: Use the defined generator in step 3 for training set and validation set
# 
# Here I feed for the generator both the training and validation samples.

# In[5]:


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
print("Done")


# This is an image augmentation function which I did use.
# The function flipps the image then changes its brightness.
# 
# Flipping is done using np.fliplr which flips the imges horizonally (like a mirror)
# Changing the brightness is done by converting the image from RGB to HSV  (Hue, Saturation, Value) space, in which the value is responsible on the brightness.
# The v value is changes then the image is converted back to RGB

# In[6]:


def image_augmentation(image):
    flipped_image = np.fliplr(image)
    hsv_image = cv2.cvtColor(flipped_image, cv2.COLOR_RGB2HSV) #convert it to hsv

    h, s, v = cv2.split(hsv_image)
    v += 100
    final_hsv_image = cv2.merge((h, s, v))

    aug_image = cv2.cvtColor(final_hsv_image, cv2.COLOR_HSV2RGB)
    
    return aug_image


# ---
# 
# ## Step 5: Using keras, build a regression model based on LeNet architecture to predict the steering angle
# 
# Here, I made use of the LeNet Architecture by Yan LeCun that is shown below ** with adding dropout for the fully connected layers AND modifying the output layer to be of output 1 as it is regression problem not classification**.
# 
# ![LeNet Architecture](lenet.png)
# 
# Here I will explain the model input, a brief at each layer mentioning the dimenions of each one
# 
# ### Input
# At the begging of the model, I added two pre-processing layers to the images:
# 1) Normalization and Mean-centering "scaling to have a zero-mean images" 
# 2) Cropping:
# 
# 1) **Lambda Layer:** This layer normalizes and scales the images. The lambda parallizes the processing. This layer takes each pixel of the image and applies to it:
#  * Normalization: `pixel_normalized = pixel / 255`
#  * Mean-centering: `pixel_mean_centered = pixel_normalized - 0.5`
# 
# > The input to the lambda layer is the image captured from the simulator at training which is 160 pixels high, 320 pixels wide, and 3 channels (R, G, B) 
#  
# 2) **Cropping Layer:** As the top scene of the image captures hills, trees, sky, and other elements tha may distract our model more than help. Alos, the very bottom of the image shows the hood of the car. So, it makes sense to crop these porions of the image. I am cropping 70 pixels from the top, 20 pixels from the bottom, and no cropping from left and right.
# 
# > The output of this layer should be an image of 70 pixels high, 320 pixels wide, and 3 channels
# 
# ### Architecture
# **Layer 1: Convolutional.** The output shape should be 66 h x 316 w x 6.
# 
# **Activation.** ELU "Exponential Linear Unit" activation function
# 
# **Pooling.** The output shape should be 33x158x6.
# 
# **Layer 2: Convolutional.** The output shape should be 29x154x16.
# 
# **Activation:** ELU activation function
# 
# **Pooling:** The output shape should be 15x77x16.
# 
# **Flatten:** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The output should be 18480
# 
# **Layer 3: Dense "Fully Connected" Layer.** This should have 120 outputs.
# 
# **Activation:** ELU activation function
# 
# **Dropout:** A dropout is used in order to pretent our model to memorize the training set. We get consensus opinion by averaging the activations (I used a dropout of 0.5).
# 
# **Layer 4: Dense "Fully Connected" Layer.** This should have 84 outputs.
# 
# **Activation:** ELU activation function.
# 
# **Dropout:** A dropout is used in order to pretent our model to memorize the training set. We get consensus opinion by averaging the activations (I used a dropout of 0.5)
# 
# **Layer 5: Dense "Fully Connected" Layer (Output -> Regression).** This should have 1 output of predicting the steering angle
# 
# ### Output
# Steering angle prediction
# 
# 
# > I used ELU activation function rather than ReLU as ELU function takes care of the Vanishing Gradient Problem. 
# 
# > Here I have used the mean square error loss function as it is the most convenient for regression problems. The yields a derivative cose function (convex).
# 
# > I used ADAM optimizer as it maintains a per-parameter learning rate that improves performance on problems with sparse gradients (e.g. natural language and computer vision problems). [source](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning)
# 
# At the end, I saved the model to test it on the simulator

# In[7]:


# Import Keras packes needed
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Activation, Cropping2D, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Define the model
model = Sequential()

#Preparing the input by adding lambda and cropping layers
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,20), (0,0))))

# Layer 1: Convuluation 5x5. Output is 66 h x 316 w x 6
model.add(Convolution2D(6, 5, 5, border_mode='valid'))
# MaxPooling. output is 33x158x6
model.add(MaxPooling2D((2, 2)))
model.add(Activation('elu'))

# Layer 1: Convuluation 5x5. Output is 29x154x16.
model.add(Convolution2D(16, 5, 5, border_mode='valid'))
# MaxPooling. output is 15x77x16
model.add(MaxPooling2D((2, 2)))
model.add(Activation('elu'))

# Flatten the convoltion for fully connected layer (Dense)
model.add(Flatten())

# Fully connected layer. Output is 120
model.add(Dense(120))
model.add(Activation('elu'))
model.add(Dropout(0.5))

# Fully connected layer. Output is 84
model.add(Dense(84))
model.add(Activation('elu'))
model.add(Dropout(0.5))

# Output 
model.add(Dense(1))

# Compile the model using mean square error  loss function and adam optimizer
model.compile(loss='mse', optimizer='adam')


# Use fit_generator to train and validate the model using the generators defined above.
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=7,
                    verbose=1)

# Save the model for testing it on the simulator
model.save('model.h5')

