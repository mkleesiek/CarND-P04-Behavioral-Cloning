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




To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)



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
