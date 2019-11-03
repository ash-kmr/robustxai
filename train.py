from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import animation
from IPython.display import HTML
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.ops import gen_array_ops, nn
from tensorflow.python.keras import initializers, regularizers, constraints
from tensorflow.python.framework import ops
from tensorflow.keras import layers as L
import cv2
from tensorflow.python.ops.parallel_for.gradients import jacobian
from tensorflow.keras import losses as kloss
from tensorflow.keras import Model

tf.compat.v1.enable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# class AutoEncoder(Model):
#     def __init__(self):
#         super(AutoEncoder, self).__init__()
        
class InterpretableNetwork(Model):
    def __init__(self, input_shape):
        super(InterpretableNetwork, self).__init__()
        
        # self.input_layer = L.InputLayer(input_shape=input_shape)
        
        # encoder part of the network
        self.encoder_c1 = L.Conv2D(filters=16, kernel_size=3, padding="SAME", activation="relu")
        self.encoder_m1 = L.MaxPool2D(strides=2)
        self.encoder_c2 = L.Conv2D(filters=64, kernel_size=3, padding="SAME", activation="relu")
        self.encoder_m2 = L.MaxPool2D(strides=2)
        self.encoder_f = L.Flatten()
        
        # concepts
        self.encoder_d1 = L.Dense(10, activation="relu")
        
        # decoder part of the network
        self.decoder_d1 = L.Dense(3136)
        self.decoder_r = L.Reshape((7, 7, 64))
        self.decoder_m1 = L.UpSampling2D()
        self.decoder_c1 = L.Conv2D(filters=16, kernel_size=3, padding="SAME", activation="relu")
        self.decoder_m2 = L.UpSampling2D()
        self.decoder_c2 = L.Conv2D(filters=1, kernel_size=3, padding="SAME", activation="relu")
        
        # robustness part of the network
        self.robust_c1 = L.Conv2D(filters=16, kernel_size=3, padding="SAME", activation="relu")
        self.robust_m1 = L.MaxPool2D(strides=2)
        self.robust_c2 = L.Conv2D(filters=64, kernel_size=3, padding="SAME", activation="relu")
        self.robust_m2 = L.MaxPool2D(strides=2)
        self.robust_f = L.Flatten()
        
        # weights for each concept
        self.robust_d1 = L.Dense(10, activation="tanh")
        
        # classificatoin part of the network
        self.classifier_d1 = L.Dense(10, activation="relu")
        self.classifier_d2 = L.Dense(10, activation="softmax")
        
    def call(self, x):
        """
        Overriding the call method
        """
        
        # x = self.input_layer(x)
        
        # getting the concepts from the encoder
        enc = self.encoder_c1(x)
        enc = self.encoder_m1(enc)
        enc = self.encoder_c2(enc)
        enc = self.encoder_m2(enc)
        enc = self.encoder_f(enc)
        concepts = self.encoder_d1(enc)
        
        # passing the concepts to the decoder
        dec = self.decoder_d1(concepts)
        dec = self.decoder_r(dec)
        dec = self.decoder_m1(dec)
        dec = self.decoder_c1(dec)
        dec = self.decoder_m2(dec)
        reconstructed = self.decoder_c2(dec)
        
        # getting the weights for concepts
        rob = self.robust_c1(x)
        rob = self.robust_m1(rob)
        rob = self.robust_c2(rob)
        rob = self.robust_m2(rob)
        rob = self.robust_f(rob)
        weights = self.robust_d1(rob)
        
        # weighting the concepts with weights
        weighted_concepts = tf.math.multiply(concepts, weights)
        
        # performing classification
        out = self.classifier_d1(weighted_concepts)
        out = self.classifier_d2(out)
        
        return x, concepts, reconstructed, weights, out
    
    def helper(self, x):
        
        # x = self.input_layer(x)
        
        # getting the concepts from the encoder
        enc = self.encoder_c1(x)
        enc = self.encoder_m1(enc)
        enc = self.encoder_c2(enc)
        enc = self.encoder_m2(enc)
        enc = self.encoder_f(enc)
        concepts = self.encoder_d1(enc)
        
        # passing the concepts to the decoder
        dec = self.decoder_d1(concepts)
        dec = self.decoder_r(dec)
        dec = self.decoder_m1(dec)
        dec = self.decoder_c1(dec)
        dec = self.decoder_m2(dec)
        reconstructed = self.decoder_c2(dec)
        
        # getting the weights for concepts
        rob = self.robust_c1(x)
        rob = self.robust_m1(rob)
        rob = self.robust_c2(rob)
        rob = self.robust_m2(rob)
        rob = self.robust_f(rob)
        weights = self.robust_d1(rob)
        
        # weighting the concepts with weights
        weighted_concepts = tf.math.multiply(concepts, weights)
        
        # performing classification
        out = self.classifier_d1(weighted_concepts)
        out = self.classifier_d2(out)
        
        return x, concepts, out       


model = InterpretableNetwork(input_shape=(28, 28))
model.build((None, 28, 28, 1))
model.summary()
@tf.function
def getGrad1(image):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(image)
        x, concepts, out = model.helper(image)
    gradients = tape.gradient(out, x)
    return gradients

@tf.function
def getGrad2(image):
    with tf.GradientTape(persistent=True) as tape:
        x, concepts, _ = model.helper(image)
    gradients = jacobian(concepts, x)
    return gradients
    
@tf.function
def RobustnessLoss(image, weights):
    a = getGrad1(image)
    b = getGrad2(image)
    temp = tf.expand_dims(tf.expand_dims(tf.expand_dims(weights, axis=2), axis=3), axis=4)
    b = tf.reduce_sum(b, axis=2)
    temp = tf.broadcast_to(temp, b.shape)
    toreturn = tf.reduce_sum(tf.math.multiply(temp, b), axis=1)
    toreturn = tf.reshape(toreturn, [toreturn.shape[0], -1])
    a = tf.reshape(a, [a.shape[0], -1])
    return tf.reduce_mean(tf.norm(a-toreturn, axis=1), axis=0)

@tf.function
def reconstruction_loss(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.losses.mean_squared_error(y_true, y_pred))

classification_loss = kloss.SparseCategoricalCrossentropy()
train_reconstruction = tf.keras.metrics.Mean(name='reconstruction')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_rob = tf.keras.metrics.Mean(name='train_rob')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(image, labels):
    with tf.GradientTape() as tape:
        _, concepts, reconstructed, weights, prediction = model(image)
        loss_rec = reconstruction_loss(reconstructed, image)
        loss_class = classification_loss(labels, prediction)
        loss_rob = RobustnessLoss(image, weights)
        loss = loss_rec + loss_class + loss_rob
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_reconstruction(loss_rec)
    train_loss(loss_class)
    train_rob(loss_rob)
    train_accuracy(labels, prediction)

import tensorflow_datasets as tfds
dataset = tfds.load(name="mnist", split=tfds.Split.TRAIN)
dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
EPOCHS = 1000

for epoch in range(EPOCHS):
    bt = 0
    for image in dataset:
        img = image['image']
        label = image['label']
        castedImage = tf.keras.backend.cast(img, 'float32')
        train_step(castedImage, label)
    template = 'Epoch {}, R. Loss: {}, C. Loss: {}, R.Loss: {}, Acc = {}'
    print(template.format(epoch+1,
                        train_reconstruction.result(),
                        train_loss.result(),
                        train_rob.result(),
                        train_accuracy.result()))

    # Reset the metrics for the next epoch
    train_reconstruction.reset_states()
    train_loss.reset_states()
    train_rob.reset_states()
    train_accuracy.reset_states()