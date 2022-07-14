import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
import matplotlib.image as mimg
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, LSTM, TimeDistributed, MaxPooling1D, MaxPooling2D, MaxPooling3D, \
    Reshape, Flatten, RepeatVector, Bidirectional
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K, Sequential
from keras.applications.xception import Xception, decode_predictions

from keras.optimizers import Adam
import tensorflow as tf
from keras.layers.core import Lambda
from tensorflow.python.framework import ops
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import glob
import numpy as np
from keras.utils import multi_gpu_model
import h5py
from keras.utils.io_utils import HDF5Matrix
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.metrics import roc_curve,roc_auc_score
import random
from sklearn.metrics import confusion_matrix

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda, Input
from keras import backend as K
from keras.activations import sigmoid
from keras.utils import to_categorical

### ############################################################################################################
## Hyperparameter
## 해당 하이퍼파라미터를 설정하여 사용하시면 됩니다.

num_frame = 

img_rows = 
img_cols = 
img_size = (img_rows, img_cols)

num_filiters = 

channel = 

#############################################################################################################

def attach_attention_module(net, attention_module):
    if attention_module == 'se_block':  # SE_block
        net = se_block(net)
    elif attention_module == 'cbam_block':  # CBAM_block
        net = cbam_block(net)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net


def se_block(input_feature, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    channel = input_feature.shape[channel_axis]
    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, channel)
    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, channel // ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, channel)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature


def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    channel = input_feature.shape[channel_axis]
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 0.001
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def plot_roc_curve(fpr,tpr):
    print("fpr = [", end='')
    for i in fpr:
        print(str(i)+", ",end='')
    print("]")

    print("tpr = [", end='')
    for i in tpr:
        print(str(i) + ", ", end='')
    print("]")


def get_img_array(img_path, size):
    min_max_scaler = MinMaxScaler()

    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)

    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    #inputs_layer = keras.Input(shape=(img_rows, img_cols, 3))
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
    #last_conv_layer_model = keras.Model(inputs_layer, last_conv_layer)
    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    #preprocess_input = keras.applications.xception.preprocess_input
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def grad_cam_heatmap(
    img_array, img_path, model, last_conv_layer_name,
    classifier_layer_names, count):

    heatmap = make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names)

    # We rescale heatmap to a range 0-255
    for i in range(num_frame):
        heatmap = np.uint8(255 * heatmap)

        # We use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # We use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # We create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img_cols, img_rows))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * 0.4 + (img_array[0][i]*255)
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
        #superimposed_img.show()

        # Save the superimposed image
        save_path = img_path+str(count)+"_"+str(i)+".png"
        superimposed_img.save(save_path)

def model_xception():
    base_model = Xception(weights='imagenet', input_shape=(img_rows, img_cols, 3),
        include_top=False)
    base_model.trainable = False

    x = attach_attention_module(base_model.output, 'cbam_block')

    model_attention = Model(base_model.input, x, name='attention')

    return model_attention

def model_EfficientNet():
    base_model = tf.keras.applications.EfficientNetB7(weights='imagenet', include_top=False,
        input_shape=(img_rows, img_cols, 3))

    base_model.trainable = False

    x = attach_attention_module(base_model.output, 'cbam_block')

    model_attention = Model(base_model.input, x, name='attention')

    return model_attention

def model_cbam(shape_):
    inputs_attention = keras.Input(shape=(shape_[1], shape_[2], shape_[3]))

    x_attention = attach_attention_module(inputs_attention, 'cbam_block')

    model_attention = Model(inputs_attention, x_attention, name='attention')

    return model_attention

def model_layer(inputs):
    backbone_model = model_xception()

    x = TimeDistributed(backbone_model, name="xception")(inputs)

    x = Bidirectional(ConvLSTM2D(filters=15, kernel_size=(3, 3),
                   padding='same', return_sequences=False), name="birectional_ConvLSTM2D")(x)

    x = BatchNormalization()(x)

    x = Flatten(name='flatten')(x)
    x = Dense(2, activation='softmax', name='predictions')(x)

    return x

def make_model():
    inputs = keras.Input(shape=(num_frame, img_rows, img_cols, channel))

    x = model_layer(inputs)

    model = Model(inputs, x, name='XceptionandCBAM')

    return model

if __name__ == "__main__":
    model = make_model()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(10), beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])