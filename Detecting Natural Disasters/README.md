## Detecting Natural Disasters 

DEMO:  [![Nbviewer](https://github.com/jupyter/design/blob/main/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/shejz/Disaster-Detection/blob/main/Detecting%20Natural%20Disasters/natural_disaster_detection.ipynb)

To create the natural disaster detector we fine-tuned VGG16 (pre-trained on ImageNet) on a dataset of 4,428 images belonging to four classes:

- Cyclone/Hurricane: 928 images
- Earthquake: 1,350
- Flood: 1,073
- Wildfire: 1,077

![](https://github.com/shejz/Disaster-Detection/blob/main/Detecting%20Natural%20Disasters/natural_disaster_classes.jpg)

After the model was trained we evaluated it on the testing set, finding that it obtained **95%** classification accuracy.

### Output

![](https://github.com/shejz/Disaster-Detection/blob/main/Detecting%20Natural%20Disasters/output/keras_natural_disaster_flood.gif)

![](https://github.com/shejz/Disaster-Detection/blob/main/Detecting%20Natural%20Disasters/output/keras_natural_disaster_earthquake.gif)

![](https://github.com/shejz/Disaster-Detection/blob/main/Detecting%20Natural%20Disasters/output/keras_natural_disaster_wildfire.gif)


