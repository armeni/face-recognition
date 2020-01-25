# Face recognition 


## Description
This script implements face recognition algorithm with memorization. When a person looks at the camera for the first time, he will not be recognized. The algorithm will create a series of photos of a person. Next time this person will be identified. 

## Usage

- Run recognition.py script.
- If you want to train the model with other photos, add these photos to the 'data' directory and run extract_embeddings.py and train_model.py (Preliminary remove '#' in the last line).

## Requirements

- [OpenCV](https://opencv.org/)
- [Pickle](https://docs.python.org/3/library/pickle.html)
- [Imutils](https://github.com/jrosebr1/imutils) 
- [OpenFace](https://cmusatyalab.github.io/openface/) (models/openface_nn4.small2.v1.t7/) — is a Python and [Torch](https://pytorch.org/) implementation of face recognition with deep neural networks
- OpenCV’s [Caffe](https://caffe.berkeleyvision.org/)-based deep learning face detector used to actually localize the faces in the images (models/deploy.prototxt/ and models/res10_300x300_ssd_iter_140000.caffemodel/)
