import keras.backend as K
from keras import Sequential, Input, Model
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D

K.set_image_dim_ordering('tf')


# Create function to build the generator CNN model
def build_generator():
    gen_model = Sequential()

    gen_model.add(Dense(input_dim=100, output_dim=2048))
    gen_model.add(BatchNormalization())
    gen_model.add(ReLU())

    gen_model.add(Dense(256 * 8 * 8))
    gen_model.add(BatchNormalization())
    gen_model.add(ReLU())
    gen_model.add(Reshape((8, 8, 256), input_shape=(256 * 8 * 8,)))
    gen_model.add(UpSampling2D(size=(2, 2)))

    gen_model.add(Conv2D(128, (5, 5), padding='same'))
    gen_model.add(BatchNormalization())
    gen_model.add(ReLU())

    gen_model.add(UpSampling2D(size=(2, 2)))

    gen_model.add(Conv2D(64, (5, 5), padding='same'))
    gen_model.add(BatchNormalization())
    gen_model.add(ReLU())

    gen_model.add(UpSampling2D(size=(2, 2)))

    gen_model.add(Conv2D(3, (5, 5), padding='same'))
    gen_model.add(Activation('tanh'))
    return gen_model

# Create function to build the discriminator CNN model
def build_discriminator():
    dis_model = Sequential()
    dis_model.add(
        Conv2D(128, (5, 5),
               padding='same',
               input_shape=(64, 64, 3))
    )
    dis_model.add(BatchNormalization())
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    #dis_model.add(Conv2D(256, (3, 3)))
    #dis_model.add(BatchNormalization())
    #dis_model.add(LeakyReLU(alpha=0.2))
    #dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    dis_model.add(Conv2D(512, (3, 3)))
    dis_model.add(BatchNormalization())
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    dis_model.add(Flatten())
    dis_model.add(Dense(1024))
    dis_model.add(BatchNormalization())
    dis_model.add(LeakyReLU(alpha=0.2))

    dis_model.add(Dense(1))
    dis_model.add(Activation('sigmoid'))

    return dis_model

# Create function to build the adversarial model combining the generator and discriminator models
def build_adversarial_model(gen_model, dis_model):
    model = Sequential()
    model.add(gen_model)
    dis_model.trainable = False
    model.add(dis_model)
    return model
