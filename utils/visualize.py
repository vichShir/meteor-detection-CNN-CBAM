'''
This code was adapted from: Vittorio Mazzia
'''

# Copyright 2021 Vittorio Mazzia. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def save_img(img_array, filename):
    rescaled = (255.0 / img_array.max() * (img_array - img_array.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save(filename)


def explainGradCamCompare(i, explainer, ax, img, y, model_1, y_pred_1, model_2, y_pred_2, class_names):
    """
    [Attention Episode]
    Plot GRADCAM of TWO trained models. It needs an axes with two columns
    """
    data = ([img], None)
    image = np.squeeze(img)
    y_predm_1 = np.argmax(y_pred_1)
    y_predm_2 = np.argmax(y_pred_2)

    grid_1 = explainer.explain(
        data, model_1, class_index=y_predm_1, image_weight=0.8, layer_name='block5_pool')
    grid_2 = explainer.explain(
        data, model_2, class_index=y_predm_2, image_weight=0.8, layer_name='block5_pool')

    ax[1].set_xlabel("Pred: {} {:2.0f}% ({})".format(class_names[y_predm_1],
                                                     100*np.max(y_pred_1),
                                                     #class_names[y], 
                                                     i),
                     color=('blue' if y == y_predm_1 else 'red'))

    ax[2].set_xlabel("Pred: {} {:2.0f}% ({})".format(class_names[y_predm_2],
                                                     100*np.max(y_pred_2),
                                                     #class_names[y], 
                                                     i),
                     color=('blue' if y == y_predm_2 else 'red'))
    ax[0].imshow(image)
    ax[1].imshow(grid_1)
    ax[2].imshow(grid_2)