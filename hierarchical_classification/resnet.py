import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import numpy as np
from .classifier_template import Classifier
from tensorflow.keras.optimizers import Adam, AdamW, SGD
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

class Resnet50(Classifier):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        input_shape: tuple = (224, 224, 3)
    ) -> None:
        
        self.num_classes = num_classes
        self.input_shape = input_shape

        weights = 'imagenet' if pretrained else None
        base_model = ResNet50(weights=weights, include_top=False, input_tensor=Input(shape=self.input_shape))
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        output_tensor = Dense(self.num_classes, activation='sigmoid')(x)

        self.model = Model(inputs=base_model.input, outputs=output_tensor)
    
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
    
    def train(
        self,
        X: np.array, # corresponds to spectrograms_preprocessed in notebook
        Y: np.array,
        test_size: float = 0.2,
        optimizer: str = 'adam',
        learning_rate: float = 0.0001,
        loss: str = 'binarycrossentropy',
        metrics: list = ['accuracy'],
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        threshold: float = 0.5
    ) -> np.array:
        
        # setup
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
        optimizers = {
            'adam': Adam(learning_rate=learning_rate),
            'adamw': AdamW(learning_rate=learning_rate),
            'sgd': SGD(learning_rate=learning_rate)
        }
        loss_functions = {
            'binarycrossentropy': BinaryCrossentropy()
        }
        self.model.compile(optimizer=optimizers[optimizer], loss=loss_functions[loss], metrics=metrics)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # train
        history = self.model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping]
        )

        # predict
        predictions = self.model.predict(X)
        binary_predictions = (predictions > threshold).astype(int)

        return binary_predictions

    def to_next_classifier(
        self,
        predictions,
        spectrograms
    ) -> None:
        pass
