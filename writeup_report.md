# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the suggested sample data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

I used the sample driving data from given link as (https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). I flipped the images to get available more examples to train the model. I cropped the images to avoid unnecessary information from image or to only see section with road. Some of the images are given below.

[//]: # (Image References)

[![Normal Left Camera Image](./examples/left_sample.jpg)](./examples/left_sample.jpg)
[![Normal Right Camera Image](./examples/right_sample.jpg)](./examples/right_sample.jpg)
[![Normal Center Camera Image](./examples/center_sample.jpg)](./examples/center_sample.jpg)

<!-- [image6]: ./examples/center_flipped.jpg "Flipped Center Camera Image" -->

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py, containing the script to create and train the model
* drive.py, for driving the car in autonomous mode
* behavioral_cloning.h5, containing a trained convolution neural network 
* writeup_report.md, summarizing the results
* video_output.mp4, the video output from the created images by simulator with driving the car in autonomous mode which is used by video.py to create video

#### 2. Submission includes functional code
Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py behavioral_cloning.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I took the model from nvidia paper. My model consists of a convolution neural network 5x5 and  3 X 3 filter sizes and depths between 24 and 64 (model.py lines 89-114). There is 3 fully connected layers are used and of size 100, 50 and 10. The last layer is Dense of size 1 that is giving output for steering.

[![nvidia paper snapshot](./examples/nvidia_paper_snapshot.png)](./examples/nvidia_paper_snapshot.png)

The model includes ELU layers to introduce nonlinearity(eg. line 98, 100, 102 etc.), and the data is normalized in the model using a Keras lambda layer (code line 92). 

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting, L2 regularizer is used (kernel_regularizer=regularizers.l2(0.0001) in fully connected layer of the architure  (model.py lines 108, 110 and 112).

Data was split into training and validation sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, 5 EPOCHS. Adam optimizer set the learning rate automatically to get best results. So the learning rate was not tuned manually (model.py line 116, 117). On 3 to 5 EPOCHS, it shows some improvement to learn but if EPOCHS increases, there is no positie effect on learning. So it is tuned to best on 5. Batch size is used of 128.

#### 4. Appropriate training data

I used udacity provided sample data to train the model. Training data was chosen to keep the vehicle driving on the road. Centre, left and right camera images were used for training the model. The training data was also augmented by flipping the images.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create a successful model which is able to drive the car automatically on given track on provided simulator.

My first step was to use a convolution neural network model similar to the nvidia pipeline paper that is suggested in class lessons. I thought this model might be appropriate because nvidia successfully drove the driverless car with this model.

In order to gauge how well the model was working, first I run my model with center image and angle only. It was not satisfactory. After that I took left and right images and correct the angle with correction value (0.2). Still, the result on simulator was not satisfactory. I changed the correction values to 0.1 and 0.3 and see the results. The car once goes out of the road and run in the clockwise circle and anti-clockwise on change the correction value of angles but not in the right direction.

I thought, there is no sufficient data to train the model, so I flip all (center, left and right) images and change their angles negatively as well. Now I have double data as compare to existing data. I used croped images with this to optimise the results.

To combat the overfitting, I modified the model and added the L2 kernel regularizer (kernel_regularizer=regularizers.l2(0.0001)) in the fully connected layer of the model.

Then I found that my model is trained good enough. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 89-114) consisted of a convolution neural network with the following layers and layer sizes.

Final model architecture:

| Layer         						|     Description	        						| 
|:---------------------:				|:---------------------------------------------:	| 
| Lambda layer (preprocess-input)		| Normalized, centered around zero with small standard deviation	| 
| Cropping2D Layer (preprocess-input)	| 160x320 RGB image   								| 
| Convolution Layer     				| 24 channels, 5x5 kernel, 2x2 stride, valid padding 						|
| ELU									| Activation Layer									|
| Convolution Layer     				| 36 channels, 5x5 kernel, 2x2 stride, valid padding 						|
| ELU									| Activation Layer									|
| Convolution Layer     				| 48 channels, 5x5 kernel, 2x2 stride, valid padding 						|
| ELU									| Activation Layer									|
| Convolution Layer     				| 64 channels, 3x3 kernel, 1x1 stride, valid padding 						|
| ELU									| Activation Layer									|
| Convolution Layer     				| 64 channels, 3x3 kernel, 1x1 stride, valid padding 						|
| ELU									| Activation Layer									|
| Flatten Layer	      					| Flattening output of previous layer 				|
| Fully Connected Layer 	    		| Fully connected layer of size 100, with L2 kernel regularizer 				|
| ELU									| Activation Layer									|
| Fully Connected Layer 	    		| Fully connected layer of size 50, with L2 kernel regularizer 					|
| ELU									| Activation Layer									|
| Fully Connected Layer 	    		| Fully connected layer of size 10, with L2 kernel regularizer 					|
| ELU									| Activation Layer									|
| Fully Connected Layer (final)	    	| Fully connected final layer 						|


#### 3. Training Set Data & Training Process

I used udacity provided data to train model about good driving behavior. Here is an example image of center lane driving:

![center camera image][image2]

The right camera sample image:

![alt text][image3]

The left camera sample image:

![alt text][image4]


To augment the data set, I also flipped images and angles thinking that this would add more examples to train the model well as more data is better to train model better. For example, here is an image that has then been flipped:

![center flipped][image6]
![left flipped][image7]
![right flipped][image7]

After the collection process, I had 3 times more number of data points. I then preprocessed incoming data by normalize and cropping the image, centered around zero with small standard deviation (x/127.5 - 1.0) and crop image to only see section with road. Images was cropped 50px from top and 20px from bottom. All the preprocessing was done in the start model itself.

I get randomly shuffled data set and put 20% of the data into a validation set(model.py line 29). I used the python generator that provide the batches of data for training and validation of the model. (model.py lines 32, 82)

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as I used 3 to 7 and found that 5 is good to use. Till the size of epochs is 5, the training and validation shows good results but after that it stops improving. I used an adam optimizer so that manually training the learning rate wasn't necessary.
