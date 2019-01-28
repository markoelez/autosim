# Python Autonomous Car with Tensorflow

Explores the capabilities of Python to play Udacity's open source Self-Driving Car Simulator with an autonomous agent built with deep learning techniques.

This program does not use Udacity's built in frame recording function. Rather, it reads frames in directly from the desktop using OpenCV.

In order setup the program correctly, the Udacity simulator should be set to a screen resolution of 640x800 and positioned in the upper left corner of your desktop as shown below:

![image](https://github.com/markoelez/pyAutoSim-mac/blob/master/example_config.png)

# Usage:

1. generate training data by running `main.py'
2. balance trainging data by running `balance_data.py`
3. train model by running `train_model.py`
4. run `test_model.py` to test your model in the virtual environment

To add custom models, simply define them in a function in `models.py`
