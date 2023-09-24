import tensorflow as tf
from keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                          MaxPooling2D, Flatten, Dense, Concatenate, Dropout)

import numpy as np
import os


class PoseDetection:
    def __init__(self, data_folder, target_size=(224, 224), num_keypoints=24):
        self.data_folder = data_folder
        self.target_size = target_size
        self.num_keypoints = num_keypoints
        self.model = self._build_model()

    def _build_model(self):
        # Define the first input layer for images
        input_images = Input(shape=(self.target_size[0], self.target_size[1], 3))

        # Define the second input layer for keypoints
        input_keypoints = Input(shape=(24,))

        # Convolutional Block 1 for images
        x_images = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_images)
        x_images = BatchNormalization()(x_images)
        x_images = Activation('relu')(x_images)
        x_images = MaxPooling2D(pool_size=(2, 2))(x_images)

        # Convolutional Block 2 for images
        x_images = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(x_images)
        x_images = BatchNormalization()(x_images)
        x_images = Activation('relu')(x_images)
        x_images = MaxPooling2D(pool_size=(2, 2))(x_images)

        # Convolutional Block 3 for images
        x_images = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x_images)
        x_images = BatchNormalization()(x_images)
        x_images = Activation('relu')(x_images)
        x_images = MaxPooling2D(pool_size=(2, 2))(x_images)

        # Flatten the output of the image processing
        x_images = Flatten()(x_images)

        # Fully Connected Layers for the image branch
        x_images = Dense(512, activation='relu')(x_images)
        x_images = Dropout(0.5)(x_images)  # Regularization
        x_images = Dense(256, activation='relu')(x_images)

        # Concatenate the image features with the keypoints
        combined_features = Concatenate()([x_images, input_keypoints])

        # Fully Connected Layers for the combined features
        x = Dense(512, activation='relu')(combined_features)
        x = Dropout(0.5)(x)  # Regularization
        x = Dense(256, activation='relu')(x)

        # Output layer for keypoints
        keypoints = Dense(24, activation='linear', name='keypoints_output')(x)

        model = tf.keras.Model(inputs=combined_features, outputs=[keypoints])
        model.compile(
            loss={'keypoints_output': 'mean_absolute_error'},
            optimizer='adam',
            metrics={'keypoints_output': 'mae'}
        )

        return model

    def train_model(self, train_images, train_keypoints, batch_size=1,
                    validation_split=0.1, num_epochs=1):
        self.load_model()

        # Concatenate the inputs into a single tensor
        combined_features = Concatenate()([train_keypoints, train_images])

        # Create a dictionary to specify the input names
        input_dict = {
            'input_1': train_images,  # Assuming the first input is for images
            'input_2': train_keypoints  # Assuming the second input is for keypoints
        }

        self.model.fit( train_images,  # Use the dictionary to specify inputs
                    epochs=num_epochs, batch_size=batch_size, validation_split=validation_split)

        self.save_model()

    def evaluate_model(self, test_images, test_keypoints, batch_size=1):
        # Evaluate the model and capture the evaluation metrics
         evaluation_metrics = self.model.evaluate(
             test_images,
             {'keypoints_output': test_keypoints},
             batch_size=batch_size
         )
         keypoints_predictions = self.model.predict(test_images, batch_size=batch_size)

         return keypoints_predictions

    def picture_model(self, test_images, batch_size=32):
        # Predict keypoints for the test images
        keypoints_predictions = self.model.predict(test_images, batch_size=batch_size)

        return keypoints_predictions

    def save_model(self):
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        self.model.save('saved_models/pose_model')
        print("model saved")

    def load_model(self):
        if os.path.exists('saved_models/pose_model'):
            self.model = tf.keras.models.load_model('saved_models/pose_model')
            print("model loaded")
        else:
            print("No saved model found.")

    def save_weights(self):
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        self.model.save_weights('saved_models/pose_model_weights.h5')
        print("Training weights saved")

    def load_weights(self):
        if os.path.exists('saved_models/pose_model_weights.h5'):
            self.model.load_weights('saved_models/pose_model_weights.h5')
            print("Training weights loaded")
        else:
            print("No saved weights found.")
