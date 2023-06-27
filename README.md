
# Pointbased Registaion

This repository gives an example of point based registrations for nii.gz files. Points can be computed by a instance segmentation and/or sub-segmentations. 

![Example](example.jpg)

## Install the package
#### Make venv:
```
conda create -n point_registration python=3.10 
conda install -c simpleitk -c anaconda -c conda-forge nibabel jupyter simpleitk pillow pyparsing matplotlib
pip install antspyx
```
