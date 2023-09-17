import tensorflow as tf
input_data = 0
model = tf.saved_model.load("THUNDER")

predictions = model(input_data)
