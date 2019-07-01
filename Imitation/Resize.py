from PIL import Image
import os

path = "C:/Users/Ian/PycharmProjects/Handwriting/Letters" # path to get images from
dirs = os.listdir(path)

path2 = "C:/Users/Ian/PycharmProjects/Handwriting/Letters-Resized/" # path to save images to

img_width = 200 # width of new image
img_height = 200 # heigth of new image
dim = (img_width, img_height)

for file_name in dirs: # for each file in the given directory
  image = Image.open(os.path.join(path, file_name)) # open image

  output = image.resize(dim, Image.ANTIALIAS) # resize image

  output_file_name = os.path.join(path2, "resized_" + file_name) # add "resized_" to beginning of output file name
  output.save(output_file_name, "PNG", quality=95) # save resized image in second directory