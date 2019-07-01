# adapted from code at https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py

from __future__ import print_function, division, absolute_import
import cv2
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
# from keras.utils import plot_model

def get_image_array(path): # extract pixel values of image(s) in a directory
    dirs = os.listdir(path) # get directory of images
    imgs = []
    for file_name in dirs: # for each image in the directory
        img = cv2.imread(os.path.join(path, file_name)) # read in image
        if img is not None:
            imgs.append(img) # add image to end of list
    img_array = np.expand_dims(imgs, axis=3) # add dimension to fourth axis (used for input to GAN)
    img_array = img_array[:,:,:,:,0] # get rid of last axis in the array (used for input to GAN)
    return np.asarray(img_array) # return array of pixel values from image

# extract pixel values of training images for each letter
a_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/a/")
b_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/b/")
c_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/c/")
d_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/d/")
e_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/e/")
f_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/f/")
g_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/g/")
h_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/h/")
i_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/i/")
j_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/j/")
k_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/k/")
l_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/l/")
m_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/m/")
n_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/n/")
o_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/o/")
p_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/p/")
q_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/q/")
r_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/r/")
s_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/s/")
t_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/t/")
u_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/u/")
v_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/v/")
w_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/w/")
x_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/x/")
y_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/y/")
z_imgs = get_image_array(path="C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/z/")

# set default images dimensions
img_width = 200
img_height = 200

class GAN():
    def __init__(self, train, image_path, epochs_input):
        self.X_train = train # array of pixel values
        self.path = image_path # path to save generator sample images during training
        self.img_rows = img_height # 10% of original image size
        self.img_cols = img_width # 10% of original image size
        self.channels = 1 # one channel because the images are grayscale
        self.img_shape = (self.img_rows, self.img_cols, self.channels) # img_shape = (28,28,1)
        self.latent_dim = 100 # dimension for noise vector
        self.d_loss = None # used to save discriminator loss
        self.g_loss = None # used to save generator loss
        self.d_loss_values = [] # used to plot discriminator loss
        self.d_acc_values = [] # used to plot discriminator accuracy
        self.g_loss_values = [] # used to plot generator loss
        self.epochs = epochs_input # number of training epochs

        optimizer = Adam(0.00055, 0.5) # learning rate, beta

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
        # model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
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
        # model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)
        # plot_model(model, to_file="discriminator_model.png", show_shapes=True, show_layer_names=True)

        return Model(img, validity)

    def train(self, epochs, batch_size=75, sample_interval=1000):
        # Rescale images -1 to 1
        X_train = self.X_train / 127.5 - 1.

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake =  np.zeros((batch_size, 1))

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
            self.d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            self.d_loss_values.append(self.d_loss[0]) # save discriminator loss value
            self.d_acc_values.append(self.d_loss[1]) # save discriminator accuracy value

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            self.g_loss = self.combined.train_on_batch(noise, valid)
            self.g_loss_values.append(self.g_loss) # save generator loss value

            # Plot the progress
            # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, self.d_loss[0], 100*self.d_loss[1], self.g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch): # save sample images of generator throughout the training process
        r, c = 2,2
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
        fig.savefig(self.path + "%d" % epoch + ".png")
        plt.close()

    def plot_values(self, letter): # plot discriminator loss, discriminator accuracy and generator loss values
        plt.plot(list(range(1, self.epochs+1)), self.d_loss_values, '-r', label="Discriminator Loss")
        plt.plot(list(range(1, self.epochs+1)), self.d_acc_values, '-b', label="Discriminator Accuracy")
        plt.plot(list(range(1, self.epochs+1)), self.g_loss_values, '-g', label="Generator Loss")
        plt.legend(loc="upper left")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Letter A Training Values")
        plt.savefig("C:/Users/Ian/PycharmProjects/Handwriting/Training-plots/%s.png" % letter)


if __name__ == '__main__':
    a_gan = GAN(train=a_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/a-test/images", epochs_input=100000)
    a_gan.train(epochs=100000)
    a_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_a.h5")
    a_gan.plot_values(letter="a")
    del a_gan
    b_gan = GAN(train=b_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/b-test/images", epochs_input=100000)
    b_gan.train(epochs=100000)
    b_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_b.h5")
    b_gan.plot_values(letter="b")
    del b_gan
    c_gan = GAN(train=c_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/c-test/images", epochs_input=100000)
    c_gan.train(epochs=100000)
    c_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_c.h5")
    c_gan.plot_values(letter="c")
    del c_gan
    d_gan = GAN(train=d_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/d-test/images", epochs_input=100000)
    d_gan.train(epochs=100000)
    d_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_d.h5")
    d_gan.plot_values(letter="d")
    del d_gan
    e_gan = GAN(train=e_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/e-test/images", epochs_input=100000)
    e_gan.train(epochs=100000)
    e_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_e.h5")
    e_gan.plot_values(letter="e")
    del e_gan
    f_gan = GAN(train=f_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/f-test/images", epochs_input=100000)
    f_gan.train(epochs=100000)
    f_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_f.h5")
    f_gan.plot_values(letter="f")
    del f_gan
    g_gan = GAN(train=g_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/g-test/images", epochs_input=100000)
    g_gan.train(epochs=100000)
    g_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_g.h5")
    g_gan.plot_values(letter="g")
    del g_gan
    h_gan = GAN(train=h_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/h-test/images", epochs_input=100000)
    h_gan.train(epochs=100000)
    h_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_h.h5")
    h_gan.plot_values(letter="h")
    del h_gan
    i_gan = GAN(train=i_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/i-test/images", epochs_input=100000)
    i_gan.train(epochs=100000)
    i_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_i.h5")
    i_gan.plot_values(letter="i")
    del i_gan
    j_gan = GAN(train=j_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/j-test/images", epochs_input=100000)
    j_gan.train(epochs=100000)
    j_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_j.h5")
    j_gan.plot_values(letter="j")
    del j_gan
    k_gan = GAN(train=k_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/k-test/images", epochs_input=100000)
    k_gan.train(epochs=100000)
    k_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_k.h5")
    k_gan.plot_values(letter="k")
    del k_gan
    l_gan = GAN(train=l_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/l-test/images", epochs_input=100000)
    l_gan.train(epochs=100000)
    l_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_l.h5")
    l_gan.plot_values(letter="l")
    del l_gan
    m_gan = GAN(train=m_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/m-test/images", epochs_input=100000)
    m_gan.train(epochs=100000)
    m_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_m.h5")
    m_gan.plot_values(letter="m")
    del m_gan
    n_gan = GAN(train=n_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/n-test/images", epochs_input=100000)
    n_gan.train(epochs=100000)
    n_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_n.h5")
    n_gan.plot_values(letter="n")
    del n_gan
    o_gan = GAN(train=o_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/o-test/images", epochs_input=100000)
    o_gan.train(epochs=100000)
    o_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_o.h5")
    o_gan.plot_values(letter="o")
    del o_gan
    p_gan = GAN(train=p_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/p-test/images", epochs_input=100000)
    p_gan.train(epochs=100000)
    p_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_p.h5")
    p_gan.plot_values(letter="p")
    del p_gan
    q_gan = GAN(train=q_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/q-test/images", epochs_input=100000)
    q_gan.train(epochs=100000)
    q_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_q.h5")
    q_gan.plot_values(letter="q")
    del q_gan
    r_gan = GAN(train=r_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/r-test/images", epochs_input=100000)
    r_gan.train(epochs=100000)
    r_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_r.h5")
    r_gan.plot_values(letter="r")
    del r_gan
    s_gan = GAN(train=s_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/s-test/images", epochs_input=100000)
    s_gan.train(epochs=100000)
    s_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_s.h5")
    s_gan.plot_values(letter="s")
    del s_gan
    t_gan = GAN(train=t_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/t-test/images", epochs_input=100000)
    t_gan.train(epochs=100000)
    t_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_t.h5")
    t_gan.plot_values(letter="t")
    del t_gan
    u_gan = GAN(train=u_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/u-test/images", epochs_input=100000)
    u_gan.train(epochs=100000)
    u_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_u.h5")
    u_gan.plot_values(letter="u")
    del u_gan
    v_gan = GAN(train=v_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/v-test/images", epochs_input=100000)
    v_gan.train(epochs=100000)
    v_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_v.h5")
    v_gan.plot_values(letter="v")
    del v_gan
    w_gan = GAN(train=w_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/w-test/images", epochs_input=100000)
    w_gan.train(epochs=100000)
    w_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_w.h5")
    w_gan.plot_values(letter="w")
    del w_gan
    x_gan = GAN(train=x_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/x-test/images", epochs_input=100000)
    x_gan.train(epochs=100000)
    x_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_x.h5")
    x_gan.plot_values(letter="x")
    del x_gan
    y_gan = GAN(train=y_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/y-test/images", epochs_input=100000)
    y_gan.train(epochs=100000)
    y_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_y.h5")
    y_gan.plot_values(letter="y")
    del y_gan
    z_gan = GAN(train=z_imgs, image_path="/Users/Ian/PycharmProjects/Handwriting/Letter-tests/z-test/images", epochs_input=100000)
    z_gan.train(epochs=100000)
    z_gan.generator.save("C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_z.h5")
    z_gan.plot_values(letter="z")
    del z_gan
