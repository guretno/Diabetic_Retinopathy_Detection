# Project Title

Federated learning - Institutionally Distributed Deep Learning Networks

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

Run the following commands

install the dependencies
```
pip-3.6 install -r requirement.txt
```

## Running Modes

### Preprocessing

* rescale substract (ben graham, preprocess subtract mean)
* resizes images (crop and resize)
* preprocess images (find balck image, write label master csv)
* rotate images (rotate and mirror)
* eda (if necessary, to visualize binary class, show class imbalanced)
* reconcile labels (create label train csv from mirrored and rotatedt images)
* filter labels (remove right eye and label 1)
* image to array (convert jpeg to numpy array)


### Training

* cnn (run the ml training processing)

