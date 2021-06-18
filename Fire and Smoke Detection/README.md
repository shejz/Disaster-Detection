## Fire and smoke detection with Keras and Deep Learning

NOTEBOOK:  [![Nbviewer](https://github.com/jupyter/design/blob/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/shejz/Fire-and-smoke-detection/blob/main/fire_detection.ipynb)

The dataset consisting of two classes:
- Fire
- Non-fire

To build our smoke and fire detector we utilized two datasets:

- A dataset of fire/smoke examples (1,315 images)
- A dataset of non-fire/non-smoke examples (2,688 images) containing examples of 8 natural outdoor scenes (forests, coastlines, mountains, open country, etc.). This dataset was originally put together by Oliva and Torralba for their 2001 paper, [Modeling the shape of the scene: a holistic representation of the spatial envelope](https://people.csail.mit.edu/torralba/code/spatialenvelope/).

### The 8-scenes dataset

**Gautam’s fire dataset with the 8-scenes natural image dataset so that we can classify Fire vs. Non-fire**

![](https://github.com/shejz/Fire-and-smoke-detection/blob/main/8-scenes.jpg)


The dataset we’ll be using for Non-fire examples is called 8-scenes as it contains 2,688 image examples belonging to eight natural scene categories (all without fire):
1. Coast
2. Mountain
3. Forest
4. Open country
5. Street
6. Inside city
7. Tall buildings
8. Highways


### Implementing our fire detection Convolutional Neural Network

**FireDetectionNet** — a Convolutional Neural Network for smoke and fire detection. This network was trained on our two datasets. Once our network was trained we evaluated it on our testing set and found that it obtained 92% accuracy.

**This network utilizes depthwise separable convolution rather than standard convolution as depthwise separable convolution**:

- Is more efficient, as Edge/IoT devices will have limited CPU and power draw.
- Requires less memory, as again, Edge/IoT devices have limited RAM.
- Requires less computation, as we have limited CPU horsepower.
- Can perform better than standard convolution in some cases, which can lead to a better fire/smoke detector.


### Libraries

- **matplotlib** : For generating plots with Python. Line 3 sets the backend so we can save our plots as image files.
- **tensorflow.keras** : Our TensorFlow 2.0 imports including data augmentation, stochastic gradient descent optimizer, and one-hot label encoder.
- **sklearn** : Two imports for dataset splitting and classification reporting.
**LearningRateFinder** : A class we will use for finding an optimal learning rate prior to training. When we operate our script in this mode, it will generate a plot for us to **(1)** manually inspect and **(2)** insert the optimal learning rate into our configuration file.
**FireDetectionNet** : The fire/smoke Convolutional Neural Network (CNN) that we built in the previous section.
**config** : Our configuration file of settings for this training script (it also contains settings for our prediction script).
**paths** : Contains functions from my imutils package to list images in a directory tree.
**argparse** : For parsing command line argument flags.
**cv2** : OpenCV is used for loading and preprocessing images.


### Sample Output

![](https://github.com/shejz/Fire-and-smoke-detection/blob/main/keras_fire_detection_animation.gif)


### Limitations and drawbacks
Our results are not perfect, however. Here are a few examples of incorrect classifications:

![](https://github.com/shejz/Fire-and-smoke-detection/blob/main/incorrect%20detection.jpg)

The image on the left in particular is troubling — a sunset will cast shades of reds and oranges across the sky, creating an “inferno” like effect. It appears that in those situations our fire detection model will struggle considerably.


**So, why are these incorrect classifications?**

- First, we only worked with image data. Smoke and fire can be better detected with video as fires start off as a smolder, slowly build to a critical point, and then erupt into massive flames.
- Secondly, our datasets are small. Combining the two datasets we only had a total of 4,003 images. Fire and smoke datasets are hard to come by, making it extremely challenging to create high accuracy models.

Finally, our datasets are not necessarily representative of the problem. Many of the example images in our fire/smoke dataset contained examples of professional photos captured by news reports. Fires don’t look like that in the wild. In order to improve our fire and smoke detection model, we need better data.




