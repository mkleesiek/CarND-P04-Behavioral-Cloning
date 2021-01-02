# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project of the Udacity [Self-Driving Car NanoDegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) program, I implement a Convolutional Neural Network (CNN) in [Keras](https://keras.io/) and train it to enable autonomous driving by predict steering angles from camera images in a self-driving car simulator.

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Resources
* [Self-Driving Car NanoDegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) course description at Udacity
* [Behavioral Cloning Project](https://github.com/udacity/CarND-Behavioral-Cloning-P3) template on Github
* [Udacity's Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) on Github


## Summary

The [model.py](model.py) script contains the commented code for training and saving the convolution neural network. When executed on the command line, it will provide instructions on supported command line arguments, such as the directory containing training data and several hyper parameters.

The script to drive the car [drive.py](drive.py) has not been modified from its original version provided by Udacity - all image preprocessing steps are contained within the Keras model itself.


### Model Architecture and Training Strategy

#### 1. Model Architecture

<img src="data/model.png" align="right" width="200" />

My model is based on Nvidida's 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.


## Dependencies
The implementation of this project was performed in a [Conda](https://docs.conda.io/projects/conda/en/latest/) lab environment provided by Udacity:
* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

## Structure

* [README.md](README.md): this file, including the project writeup
* [model.py](model.py): Python script for training and saving the Keras model
* [drive.py](drive.py): Python script connecting to the simulator and driving the car autonomously based on the trained model
* [video.py](video.py): Python script for creating MP4 videos when driving autonomously
* [data/model.h5](data/model.h): The trained Keras model in HDF5 format
* [data/track1.mp4](data/track1.mp4): Video recording of the simulator driving around track 1 autonomously
* [data/track2.mp4](data/track2.mp4): Video recording of the simulator driving around track 2 autonomously
* [examples/*](examples): Supplemental images for this writeyp
* [environment.yml](environment.yml]): Conda environment definition file based on the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit), but with more recent versions of TensorFlow and Keras


## License
The contents of this repository are covered under the [MIT License](LICENSE).
