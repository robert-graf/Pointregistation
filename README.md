
# Pointbased Registration
[![DOI](https://zenodo.org/badge/658692903.svg)](https://zenodo.org/badge/latestdoi/658692903)


This repository gives an example of point-based registrations for nii.gz files. Points can be computed by instance segmentation and/or sub-segmentations. 

![Example](example.jpg)

See tutorial_pointregistation.ipynb for a tutorial

## Install the package
#### Make venv:
```
conda create -n point_registration python=3.10 
conda install -c simpleitk -c anaconda -c conda-forge nibabel jupyter simpleitk pillow pyparsing matplotlib
pip install antspyx
```

# Alternative
You can use our [TPTBox](https://github.com/Hendrik-code/TPTBox/tree/main) to use this registration.
You can compute points 
```python
# Options to compute Points from segmentation
from TPTBox import calc_centroids,calc_poi_from_subreg_vert,calc_poi_from_two_segs
# Object representing pois that can be resampled to other spacings
from TPTBox import POI
from TPTBox import to_nii
poi_obj1 = calc_centroids(to_nii("path-to-seg",seg=True))
poi_obj2 = calc_centroids(to_nii("path-to-other-seg",seg=True))

from TPTBox.registration.ridged_points import Point_Registration

reg = Point_Registration(poi_obj1,poi_obj2)

nii = reg.transform_nii(to_nii("path-to-moving-image",seg=False))
nii.save(...)
poi_moved = reg.transform_poi(poi_obj2)
nii.save(...)
```
