# **Driving Behavior Cloning**
---
## Deep Learning
**Build a deep learning model the clones the driving behavior**

**In this project I have worked on a deep learning model based on LeNet architecture by Yan LeCun to clone the driving behavior of a car.**
**I have tested this model on Udacity simulator for a complete lap**

**In the following  writeup, I will be explaining each part of the project**

---
## Project Stepts:

**Step 0:** Use the simulator to collect data for good driving behavior

**Step 1:** Load The data

**Step 2:** Split data into trainig and validation sets
    
**Step 3:** Define a generator function to be used through training

**Step 4:** Use the defined generator in step 3 for training set and validation set
    
**Step 5:** Using keras, build a regression model based on LeNet architecture to predict the steering angle
    
In the next, I will be giving some details related to each steps

## Environment:
---
* AWS carnd Instance
* Python 3.6.4
* Anaconda 4.4.10


## Step 0: Use the simulator to collect data for good driving behavior
---
I have used the simulator to collect the good driving behavior.
I have collected:
 * One lap in the first track in the normal direction
 *  Another lap in the first track in the opposite direction
 *  One lap in the second track

During my data collection, I was keen to be smooth in curves. The second lap in the first track and the lap in the second track helped to have better generalized data and better performance in curves

The total dataset was 10559 examples.

## Step 1: Load The data
---
Here I am using the python csv library to read the csv file generated from the simulator.
This file contains paths to three camera images captured through the training, steering angle, throttle, break, and vehicle speed.

We will ignore the throttle, break, and speed measurements.

We will use the images as the feature set and the steering angle as the label set.

Here is the code of loading the data from the csv file:
```python
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
```

## Step 2: Split data into trainig and validation sets
---
Here I am splitting the data into two set, training set which is 80% of the data, and validation set which is 20% of the data.
The follwoing is the python code:

```python
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(len(train_samples))
print(len(validation_samples))
print("Done")
```
Training data = 8447 samples
Validation data = 2112 samples

## Step 3: Define a generator function to be used through training and validation
---
Generator is a powerful tool in python. It is used to pull pieces of the data instead of loading it at once in memory. Here in our case, we will be using generators for pre-processing data, surey pieces of them, processing them on the fly only when we need.

Here is the steps implemented inside our generator function:
 * First of all, all the steps are encapsulated in a while(1) loop in roder to prevent the generator from termination. We need the generator alive as long as the training is alive
 * Everytime, we shuffle the data. **Why** ? *In order not to make the model biased by the order of images*
 * Then we get a *slice* off our data based on the batch size defined. Here we get a batch of the training samples
 * For each batch, we get the center, left, and right camera images.
 * Also, we get the steering angle.
 * Here, I am usign a correction factor for the steering angles caputured from right and left cameras. The steering angle of the left should be less than the steering angle of the center. The steering angle of the right should be more that the steering angle of the center. Hence, I used this correction factor to compensate this difference.
 * I made a data augmentation for each image and steering angle by flipping the reading.
 * Then convert our list images and steeting angles into NumPy lists as this is the type expected by Keras.
 * Finally, before yielding (return for geterators) the feature and label sets, we shuffle.
 
Here is a sample of the generator function reading the center image and steering angle
```python
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
```

Here is a sample code handling the correction for left steering angle. For right, use -ve instead:

```python
                    center_angle = float(batch_sample[3])
                    left_angle = center_angle + correction
                    steering_angles.append(left_angle)
```

## Step 4: Use the defined generator in step 3 for training set and validation set
---
Here I feed for the generator both the training and validation samples with `batch_size = 32`.
```python
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
```

## Step 5: Using keras, build a regression model based on LeNet architecture to predict the steering angle
---
Here, I made use of the LeNet Architecture by Yan LeCun that is shown below **with adding dropout for the fully connected layers AND modifying the output layer to be of output 1 as it is regression problem not classification**.

![LeNet Architecture](https://i.imgur.com/98tBUEC.png)

Here I will explain the model input, a brief at each layer mentioning the dimenions of each one

### Input
At the begging of the model, I added two pre-processing layers to the images:
1) Lambda Layer for Normalization and Mean-centering "scaling to have a zero-mean images" 
2) Cropping Layer

**Lambda Layer:** This layer normalizes and scales the images. The lambda parallizes the processing. This layer takes each pixel of the image and applies to it:
 * Normalization: `pixel_normalized = pixel / 255`
 * Mean-centering: `pixel_mean_centered = pixel_normalized - 0.5`

> The input to the lambda layer is the image captured from the simulator at training which is 160 pixels high, 320 pixels wide, and 3 channels (R, G, B) 
 
**Cropping Layer:** As the top scene of the image captures hills, trees, sky, and other elements tha may distract our model more than help. Alos, the very bottom of the image shows the hood of the car. So, it makes sense to crop these porions of the image. I am cropping 70 pixels from the top, 20 pixels from the bottom, and no cropping from left and right.

> The output of this layer should be an image of 70 pixels high, 320 pixels wide, and 3 channels

### Model Architecure

* Make use of the LeNet Architecture **with adding dropout for the fully connected layers AND modifying the output layer to be of output 1 as it is regression problem not classification**.
  * I used the LeNet Architecture by Yan LeCun as a base
  *  **Dropout** is used in order to prevent our model to memorize the training set. We get consensus opinion by averaging the activations (I used a dropout of 0.5).
  *  I modified tha last layer to be of output 1 instead of 10 as the problem of behavior cloning is a regression problem not a classifcation prbolem

  * The following table describes the model:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Layer 1: Convolution 5x5     	| valid padding, outputs 66 h x 316 w x 6 	|
| Max pooling	      	| 2x2 stride,  outputs 33x158x6 				|
| ELU					|	Activation function							|
| Layer 2: Convolution 5x5	    | valid padding, outputs 29x154x16 |
| Max pooling	      	| 2x2 stride,  outputs 15x77x16				|
| ELU					|	Activation function							|
| Layer 3: Fully connected		| output = 120 							|
| ELU					|	Activation function						|
| Dropout					|	keep_prob = 0.5						|
| Layer 4: Fully connected		| output = 84  						|
| ELU					|	Activation function						|
| Dropout					|	keep_prob = 0.5					|
| Layer 5: Fully connected		| output = 1					|
  * The following is the python code for the model:

```python
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
```

## Conclusion
---
  * Using the LeNet Architecture with adding dropout gives outstanding training and validation accuracy.
  * Driving performance of the car on the simulator was vert acceptable and it never goes off.
  * However, we can notice that testing and validation losses are not decreasing in all times.
  * Using the [end-to-end approach by Nvidia](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) may improve these losses which may lead to better and more generalized performance.