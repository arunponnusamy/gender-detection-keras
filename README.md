# Gender detection (from scratch) using deep learning with keras and cvlib
The keras model is created by training SmallerVGGNet from scratch on around 2200 face images (~1100 for each class). Face region is cropped by applying `face detection` using `cvlib` on the images gathered from Google Images. It acheived around 96% training accuracy and ~90% validation accuracy. (20% of the dataset is used for validation)

## Python packages
* numpy
* opencv-python
* tensorflow
* keras
* cvlib

Install the required packages by executing the following command.

`$ pip install -r requirements.txt`

**Note: Python 2.x is not supported** 

Make sure `pip` is linked to Python 3.x  (`pip -V` will display this info)

Using **Python virtual environment** is highly recommended.

## Usage

### image input
`$ python detect_gender.py -i <input_image>`

### webcam
`$ python detect_gender_webcam.py`

When you run the script for the first time, it will download the pre-trained model from this [link](https://s3.ap-south-1.amazonaws.com/arunponnusamy/pre-trained-weights/gender_detection.model) and place it under `pre-trained` directory in the current path.
