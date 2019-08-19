# Handwriting-Imitation-Recognition

This repository contains the source code for a Project Lab 4 project at Texas Tech University. The program can accurately recognize handwriting from the IAM Handwriting database (http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) and imitate a user’s handwriting by displaying a picture of an inputted sentence in the user’s own handwriting. The imitation part of the program is accomplished using a neural network architecture known as a generative adversarial network to generate realistic-looking pictures of the user’s handwriting based on previously seen images. The recognition part of the program is accomplished by training a convolutional neural network on pictures of handwriting from the IAM Handwriting Database, then inputting pictures into the trained network to predict the words written in the picture. The user interacts with the program through a graphical user interface that allows the user to upload pictures for recognition and enter sentences for imitation.

The GAN code is adapted from the code at https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py

The recognition network code is adapted from the code at https://github.com/githubharald/SimpleHTR/tree/master/src
