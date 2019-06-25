# adapted from code at https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py

from __future__ import print_function, division, absolute_import

import cv2

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np
from keras.utils.vis_utils import plot_model

# set default images dimensions
img_width = 240
img_height = 80
dim = (img_width, img_height)

# resize all training images to be the same size
ian1 = cv2.imread('IanPangram1.png', cv2.IMREAD_GRAYSCALE)
ian1_resize = cv2.resize(ian1, dim, interpolation=cv2.INTER_AREA)

ian2 = cv2.imread('IanPangram2.png', cv2.IMREAD_GRAYSCALE)
ian2_resize = cv2.resize(ian2, dim, interpolation=cv2.INTER_AREA)

ian3 = cv2.imread('IanPangram3.png', cv2.IMREAD_GRAYSCALE)
ian3_resize = cv2.resize(ian3, dim, interpolation=cv2.INTER_AREA)

ian4 = cv2.imread('IanPangram4.png', cv2.IMREAD_GRAYSCALE)
ian4_resize = cv2.resize(ian4, dim, interpolation=cv2.INTER_AREA)

ian_array = [ian1_resize, ian2_resize, ian3_resize, ian4_resize]

ian_images = []
for k in range(4):
    ian_images.append(ian_array[k])
ian_images = np.expand_dims(ian_images, axis=3)
ian_images = np.asarray(ian_images)

class GAN():
    def __init__(self):
        self.img_rows = img_height # 10% of original image size
        self.img_cols = img_width # 10% of original image size
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(64, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        model.save('gen_model6.h5')
        # plot_model(model,to_file="generator_model.png", show_shapes=True, show_layer_names=True)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)
        model.save('discrim_model6.h5')
        # plot_model(model, to_file="discriminator_model.png", show_shapes=True, show_layer_names=True)

        return Model(img, validity)

    def train(self, epochs, batch_size, sample_interval):

        # # Scale training image down to 10% of original size
        # img = cv2.imread('IanPangram4.png', cv2.IMREAD_GRAYSCALE)
        # scale_percent = 10  # percent of original size
        # width = int(img.shape[1] * scale_percent / 100)
        # height = int(img.shape[0] * scale_percent / 100)
        # dim = (width, height)

        # resize image
        X_train = ian_images
        # X_train = np.expand_dims(X_train, axis=0)
        # print(X_train.shape, "\n")

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)
        # print(X_train.shape, "\n")

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 4,4
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("/Users/Ian/PycharmProjects/Handwriting/Combined-Test/images%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=40, sample_interval=100)
