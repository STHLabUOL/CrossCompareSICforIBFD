import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from tensorflow.python.ops import math_ops

class cmplxMeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
      #y_pred = tf.convert_to_tensor_v2(y_pred)
      y_true = tf.cast(y_true, y_pred.dtype)
      return tf.reduce_mean(math_ops.square(math_ops.abs(y_pred - y_true)), axis=-1)

class NeuralNetGlobalHammer(tf.keras.Model):


    def __init__(self, filterLen=32, expected_SI_power_dB=-15):
        super(NeuralNetGlobalHammer, self).__init__()

        self.filterLen = filterLen
        self.outputScaleFactor = np.sqrt(10**(expected_SI_power_dB/10))
        # Layers
        # -- Non-linear:
        self.nl1 = Conv2D(name='layerNL1', filters=8, kernel_size=(1,1), padding='valid', use_bias=False, activation='tanh', dtype=tf.float64)
        self.nl2 = Conv2D(name='layerNL3', filters=1, kernel_size=(1,1), padding='valid', use_bias=False, dtype=tf.float64)
        # -- Linear:
        self.lin_real = Conv2D(name='layerLIN1', filters=1, kernel_size=(1,filterLen), strides=1, padding='valid', use_bias=False, dtype=tf.float64)
        self.lin_imag = Conv2D(name='layerLIN2', filters=1, kernel_size=(1,filterLen), strides=1, padding='valid', use_bias=False, dtype=tf.float64)

        # Call model once with dummy input to build. This enables loading weights before first actual inputs are processed.
        # There are more elegant ways to do this (model.build()), but those didn't work for some reason.
        input_xC = tf.keras.layers.Input(name="input_x", shape=(None, None, 1), dtype=tf.complex128) # (1,None,2048)
        self(input_xC)

    
    def call(self, transmit_signal):

        input_x = transmit_signal

        # Apply MLP to magnitude
        x_mag, x_phase = tf.math.abs(input_x), tf.math.angle(input_x)
        x_mag = self.nl1(x_mag)
        x_mag = self.nl2(x_mag)

        # Apply linear model
        x_real, x_imag = x_mag*tf.math.cos(x_phase), x_mag*tf.math.sin(x_phase)
        y_hat_real = self.lin_real(x_real) - self.lin_imag(x_imag)
        y_hat_imag = self.lin_real(x_imag) + self.lin_imag(x_real)
        y_hat = tf.complex(y_hat_real, y_hat_imag)
        y_hat = self.outputScaleFactor * y_hat #scale correction to simulate learning with var=1 targets

        return y_hat


    def train_step(self, data):
        # overwrite, this function is called within model.fit()
        # this here additionally logs the latest gradient

        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Store gradients in self for access in the callback
        self.latest_gradient = gradients

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}   
    
class NeuralNetGlobalWiener(tf.keras.Model):


    def __init__(self, filterLen=32, expected_SI_power_dB=-15):
        super(NeuralNetGlobalWiener, self).__init__()

        self.filterLen = filterLen
        self.outputScaleFactor = np.sqrt(10**(expected_SI_power_dB/10))
        # Layers
        # -- Non-linear:
        self.nl1 = Conv2D(name='layerNL1', filters=8, kernel_size=(1,1), padding='valid', use_bias=True, activation='relu', dtype=tf.float64)
        self.nl2 = Conv2D(name='layerNL3', filters=1, kernel_size=(1,1), padding='valid', use_bias=True, dtype=tf.float64)
        # -- Linear:
        self.lin_real = Conv2D(name='layerLIN1', filters=1, kernel_size=(1,filterLen), strides=1, padding='valid', use_bias=False, dtype=tf.float64)
        self.lin_imag = Conv2D(name='layerLIN2', filters=1, kernel_size=(1,filterLen), strides=1, padding='valid', use_bias=False, dtype=tf.float64)

        # Call model once with dummy input to build. This enables loading weights before first actual inputs are processed.
        # There are more elegant ways to do this (model.build()), but those didn't work for some reason.
        input_xC = tf.keras.layers.Input(name="input_x", shape=(None, None, 1), dtype=tf.complex128) # (1,None,2048)
        self(input_xC)

    
    def call(self, transmit_signal):

        input_x = transmit_signal

        # Apply linear model
        x_real, x_imag = tf.math.real(input_x), tf.math.imag(input_x)
        z_hat_real = self.lin_real(x_real) - self.lin_imag(x_imag)
        z_hat_imag = self.lin_real(x_imag) + self.lin_imag(x_real)
        z_hat = tf.complex(z_hat_real, z_hat_imag)

        # Apply MLP to magnitude
        z_mag, z_phase = tf.math.abs(z_hat), tf.math.angle(z_hat)
        z_mag = self.nl1(z_mag)
        z_mag = self.nl2(z_mag)
        y_hat = tf.complex(z_mag*tf.math.cos(z_phase), z_mag*tf.math.sin(z_phase))
        y_hat = self.outputScaleFactor * y_hat #scale correction to simulate learning with var=1 targets

        return y_hat


    def train_step(self, data):
        # overwrite, this function is called within model.fit()
        # this here additionally logs the latest gradient

        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Store gradients in self for access in the callback
        self.latest_gradient = gradients

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}   

class NeuralNetGlobalHammerWiener(tf.keras.Model):
    # Combination model that hopefully can deal with both hammerstein- and wiener-data

    def __init__(self, filterLen=32, expected_SI_power_dB=-15):
        super(NeuralNetGlobalHammerWiener, self).__init__()

        self.filterLen = filterLen
        self.outputScaleFactor = np.sqrt(10**(expected_SI_power_dB/10))
        # Layers
        # -- Non-linear (1)
        self.nl1_pre = Conv2D(name='layerNL1_pre', filters=8, kernel_size=(1,1), padding='valid', use_bias=False, activation='tanh', dtype=tf.float64)
        self.nl2_pre = Conv2D(name='layerNL3_pre', filters=1, kernel_size=(1,1), padding='valid', use_bias=False, dtype=tf.float64)
        # -- Linear:
        self.lin_real = Conv2D(name='layerLIN1', filters=1, kernel_size=(1,filterLen), strides=1, padding='valid', use_bias=False, dtype=tf.float64)
        self.lin_imag = Conv2D(name='layerLIN2', filters=1, kernel_size=(1,filterLen), strides=1, padding='valid', use_bias=False, dtype=tf.float64)
        # -- Non-linear (2):
        self.nl1_post = Conv2D(name='layerNL1_post', filters=8, kernel_size=(1,1), padding='valid', use_bias=True, activation='relu', dtype=tf.float64)
        self.nl2_post = Conv2D(name='layerNL3_post', filters=1, kernel_size=(1,1), padding='valid', use_bias=True, dtype=tf.float64)

        # Call model once with dummy input to build. This enables loading weights before first actual inputs are processed.
        # There are more elegant ways to do this (model.build()), but those didn't work for some reason.
        input_xC = tf.keras.layers.Input(name="input_x", shape=(None, None, 1), dtype=tf.complex128) # (1,None,2048)
        self(input_xC)

    
    def call(self, transmit_signal):

        input_x = transmit_signal

        # Apply MLP to magnitude
        x_mag, x_phase = tf.math.abs(input_x), tf.math.angle(input_x)
        x_mag = self.nl1_pre(x_mag)
        x_mag = self.nl2_pre(x_mag)
        x_hat = tf.complex(x_mag*tf.math.cos(x_phase), x_mag*tf.math.sin(x_phase))
        
        # Apply linear model
        x_real, x_imag = tf.math.real(x_hat), tf.math.imag(x_hat)
        z_hat_real = self.lin_real(x_real) - self.lin_imag(x_imag)
        z_hat_imag = self.lin_real(x_imag) + self.lin_imag(x_real)
        z_hat = tf.complex(z_hat_real, z_hat_imag)

        # Apply MLP to magnitude
        z_mag, z_phase = tf.math.abs(z_hat), tf.math.angle(z_hat)
        z_mag = self.nl1_post(z_mag)
        z_mag = self.nl2_post(z_mag)
        y_hat = tf.complex(z_mag*tf.math.cos(z_phase), z_mag*tf.math.sin(z_phase))
        y_hat = self.outputScaleFactor * y_hat #scale correction to simulate learning with var=1 targets

        return y_hat


    def train_step(self, data):
        # overwrite, this function is called within model.fit()
        # this here additionally logs the latest gradient

        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Store gradients in self for access in the callback
        self.latest_gradient = gradients

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}   
    
def FFNN_model(chanLen, nHidden):
    # See https://github.com/abalatsoukas/fdnn
    inputLayer = Input(shape=(2*chanLen,))
    hidden1 = Dense(nHidden, activation='relu')(inputLayer)
    output1 = Dense(1, activation='linear')(hidden1)
    output2 = Dense(1, activation='linear')(hidden1)
    model = Model(inputs=inputLayer, outputs=[output1, output2])
    return model