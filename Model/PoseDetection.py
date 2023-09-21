import tensorflow as tf
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPooling2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Activation
import os


class PoseDetection:
    def __init__(self, data_folder, target_size=(224, 224), num_keypoints=12):
        self.data_folder = data_folder
        self.target_size = target_size
        self.num_keypoints = num_keypoints
        self.model = self._build_model()

    def _build_model(self):
        input_layer = tf.keras.layers.Input(shape=(self.target_size[0], self.target_size[1], 3))

        # Convolutional Block 1
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # Convolutional Block 2
        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # Convolutional Block 3
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # Global Average Pooling
        x = GlobalAveragePooling2D()(x)
        # Fully Connected Layers
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)  # Regularization
        x = Dense(256, activation='relu')(x)
        x = Dense(12, activation='linear')(x)
        # unimplemented, will work on the current functions till ready
       # pose_type = Dense(10, activation='sigmoid', name='pose_classification')

        keypoints = Dense(12, activation='linear', name='keypoints_output')(x)

      #  pose_correctness = Dense(1, activation='sigmoid', name='pose_correctness_output')(x)

        model = tf.keras.Model(inputs=input_layer, outputs=[keypoints])
        model.compile(
            loss={'keypoints_output': 'mean_absolute_error'},
            optimizer='adam',
            metrics={'keypoints_output': 'mae'}
        )

        # model.summary()
        return model

    def train_model(self, train_images, train_keypoints, batch_size=5,
                    validation_split=0.1, num_epochs=1):
        self.model.fit(train_images,
                       {'keypoints_output': train_keypoints},
                       epochs=num_epochs, batch_size=batch_size, validation_split=validation_split)
        self.save_model()

    def evaluate_model(self, test_images, test_keypoints, batch_size=32):
        # Evaluate the model and capture the evaluation metrics
        evaluation_metrics = self.model.evaluate(
            test_images,
            {'keypoints_output': test_keypoints},
            batch_size=batch_size
        )
        # Predict keypoints for the test images
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
