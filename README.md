# Representing Camera Response Function by a Single Latent Variable and Fully Connected Neural Network

Code for the paper "Representing Camera Response Function by a Single Latent Variable and Fully Connected Neural Network". If you use this code or the dataset, please cite our paper:


```
@misc{zhao2022representing,
    title={Representing Camera Response Function by a Single Latent Variable and Fully Connected Neural Network},
    author={Yunfeng Zhao and Stuart Ferguson and Huiyu Zhou and Karen Rafferty},
    year={2022},
    eprint={2209.03624},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

This repo provides an example program of the proposed model in the paper, a pre-trained camera response function (CRF) representation model, and the image dataset used. 


## File structure

```
- dataset/modified_Middlebury/    # dataset
    - CanonEOS1DsMarkII/
       | wb1i1e1-jpg.png    # jpg image
       | wb1i1e1-raw.png    # corresponding raw image
       ...
       | wb1i2e3-jpg.png
       | wb1i2e3-raw.png
       ...
    - CanonPowerShotG9/
       | wb1i1e1-jpg.png    # for calibration
       | wb1i1e1-raw.png
       ...
       | wb1i2e1-jpg.png    # for testing
       | wb1i2e1-raw.png
       ...
    ...
    | tags.json    # label information
- libs/     # other code
- model/    # pre-trained model
| example_Middlebury_calibration.py    # demonstration program
```


## Code

### Requirements

* Tensorflow 1.14.0
* opencv-python 4.4.0


### Demonstration

Run `python example_Middlebury_calibration.py` to calibrate and visualise the result.

The `example_Middlebury_calibration.py` is the demonstration program that calibrates CRF of each camera in the modified Middlebury dataset using the measured and true colour intensities of a Macbeth colour chart and compares the CRF-corrected intensities of the JPG images with the true values obtained from their raw images. 

During calibration of each camera, the program calculates the optimal parameter of the proposed model and reconstructs the CRF by the calculated model parameter and proposed model.


## Result visualisation

Camera calibration results visualised:

![Representing-crf-fig1](https://github.com/zyfccc/Representing-camera-response-function/blob/main/imgs/results.png)



## Model

In the `model` folder is a pre-trained tensorflow model using the method proposed in the paper.


## Dataset

The `dataset` folder contains the image dataset for camera calibration and result demonstration.

A list of the cameras selected in the modified Middlebury dataset:
* Canon EOS-1Ds Mark II
* Canon PowerShot G9
* Canon PowerShot G10
* Canon PowerShot S50
* Canon PowerShot S60
* Casio EX-Z55
* Nikon D70
* Nikon D200
* Olympus E10
* Olympus E500
* Panasonic DMC-LX3
* Pentax K10D
* Sony DSLR-A100
* Sony DSLR-A300

`tags.json` file provides label information that indicates regions of colour patches in the images.
