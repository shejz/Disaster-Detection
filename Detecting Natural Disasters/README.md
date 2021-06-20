## Detecting Natural Disasters 

DEMO:  [![Nbviewer](https://github.com/jupyter/design/blob/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/shejz/Disaster-Detection/blob/main/Detecting%20Natural%20Disasters/natural_disaster_detection.ipynb)

To create the natural disaster detector we fine-tuned VGG16 (pre-trained on ImageNet) on a dataset of 4,428 images belonging to four classes:

- Cyclone/Hurricane: 928 images
- Earthquake: 1,350
- Flood: 1,073
- Wildfire: 1,077

After the model was trained we evaluated it on the testing set, finding that it obtained **95%** classification accuracy.


