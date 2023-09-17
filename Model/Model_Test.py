# Shared feature extraction layers
import tensorflow as tf
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPooling2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Activation
import os


def _build_model(self):
    input_layer = tf.keras.layers.Input(shape=(self.target_size[0], self.target_size[1], 3))
    shared_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_layer)
    shared_conv = BatchNormalization()(shared_conv)
    shared_conv = Activation('relu')(shared_conv)
    shared_conv = MaxPooling2D(pool_size=(2, 2))(shared_conv)
    # ...

    # Keypoints prediction branch
    keypoints_branch = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(shared_conv)
    keypoints_branch = BatchNormalization()(keypoints_branch)
    keypoints_branch = Activation('relu')(keypoints_branch)
    keypoints_branch = MaxPooling2D(pool_size=(2, 2))(keypoints_branch)
    # Add more layers as needed
    keypoints_output = Dense(19, activation='linear', name='keypoints_output')(keypoints_branch)

    # Pose correctness branch
    pose_correctness_branch = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(shared_conv)
    pose_correctness_branch = BatchNormalization()(pose_correctness_branch)
    pose_correctness_branch = Activation('relu')(pose_correctness_branch)
    pose_correctness_branch = MaxPooling2D(pool_size=(2, 2))(pose_correctness_branch)
    # Add more layers as needed
    pose_correctness_output = Dense(1, activation='sigmoid', name='pose_correctness_output')(pose_correctness_branch)

    # Pose classification branch
    pose_classification_branch = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(shared_conv)
    pose_classification_branch = BatchNormalization()(pose_classification_branch)
    pose_classification_branch = Activation('relu')(pose_classification_branch)
    pose_classification_branch = MaxPooling2D(pool_size=(2, 2))(pose_classification_branch)
    # Add more layers as needed
    pose_classification_output = Dense(10, activation='sigmoid', name='pose_classification')(pose_classification_branch)

    model = tf.keras.Model(
        inputs=input_layer,
        outputs=[keypoints_output, pose_correctness_output, pose_classification_output]
    )

    # Compile the model with appropriate loss functions and metrics for each task
    model.compile(
        loss={
            'keypoints_output': 'mean_absolute_error',
            'pose_correctness_output': 'binary_crossentropy',
            'pose_classification': 'categorical_crossentropy'  # Adjust based on your classification task
        },
        optimizer='adam',
        metrics={
            'keypoints_output': 'mae',
            'pose_correctness_output': 'accuracy',
            'pose_classification': 'accuracy'
        }
    )
