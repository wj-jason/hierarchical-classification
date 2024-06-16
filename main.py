'''
HIGH LEVEL FUNCTIONALITY:
    - Swappable classifiers at each parent node
    - Dynamic heirarchy definitions
        - Minimal overhead when moving labels across leaf nodes
    - Distributed training/inference (?)

1. Root is a binary classifier which determines whether or not the sound is YES_CONSTRUCTION or NO_CONSTRUCTIOn
    - Since it is a multi-label problem, if an output passes the threshold for both yes and no, it will be sent to both
2. The YES_CONSTRUCTION and NO_CONSTRUCTION nodes have their own classifiers. 
    2.1. YES_CONSTRUCTION should be broken further into broad groupings 
    2.2. NO_CONSTRUCTION can immediately proceed to label predictions
3. Broad groupings from YES_CONSTRUCTION (classes TBA) similarly each have their own classifiers which proceed to label predictions
4. Aggregate results from all leaf nodes. Anything that surpassed the probability threshold will be deemed a detection. 

As of the current idea, there will only be one layer following YES_CONSTRUCTION. Hence the total # of classifiers will be:
    root + yes + no + yes_subgroups

I think there will be ~5 subgroups so ~8 classifiers total. 

ResNet50 has ~25.5M parameters, if each classifier is ResNet50, then the total architecture has ~204M parameters. 
'''

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

class ModelTree:
    def __init__(self):
        pass

    def construct_hierarchy(self):
        pass