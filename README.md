# Meteor Detection

Code used in the paper "An Optimized Training Approach for Meteor Detection with an Attention Mechanism to Improve Robustness on Limited Data".

# Environment

The experiments were run on Python 3.8.10.

To install the dependencies, run the following command:

```
pip install -r requirements.txt
```

# Getting started

We used Jupyter Notebooks to setup and run our experiments. The following were used:

1. [CNN_Model_Performance.ipynb](https://github.com/vichShir/meteor-detection-CNN-CBAM/blob/master/CNN_Model_Performance.ipynb) (trains the classification models using the 7,000 image meteor dataset)
2. [Grad-CAM_Visual_Explanations.ipynb](https://github.com/vichShir/meteor-detection-CNN-CBAM/blob/master/Grad-CAM_Visual_Explanations.ipynb) (uses the trained models to visualize the explanations with the Grad-CAM algorithm)

# License

Released under the MIT license (see [LICENSE](https://github.com/vichShir/meteor-detection-CNN-CBAM/blob/master/LICENSE))

Our code has copied or modified code from Vittorio Mazzia under the Apache-2.0 license.
