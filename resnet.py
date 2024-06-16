import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import numpy as np
from classifier import Classifier

class Resnet50(Classifier):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        input_shape: tuple = (224, 224, 3)
    ) -> None:
        
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.input_shape = input_shape

        weights = 'imagenet' if pretrained else None
        base_model = ResNet50(weights=weights, include_top=False, input_tensor=Input(shape=self.input_shape))
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        output_tensor = Dense(num_classes, activation='sigmoid')(x)

        self.model = Model(input=base_model.input, outputs=output_tensor)
    
    def resize(
        self, 
        spectrograms: np.array
    ) -> np.array:
        
        preprocessed = []
        for spec in spectrograms:
            spec = tf.expand_dims(spec, -1)
            spec_resized = tf.image.resize(spec, (224, 224))
            spec_resized = tf.image.grayscale_to_rgb(spec_resized) 
            preprocessed.append(spec_resized)
        return np.array(preprocessed)
    
    def to_next_classifier(
        self,
        predictions,
        spectrograms
    ) -> None:
        pass