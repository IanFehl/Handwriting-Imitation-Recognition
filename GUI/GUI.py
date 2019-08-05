from tkinter import *
from tkinter import filedialog
#from tkinter.ttk import *
from tkinter import messagebox
from PIL import ImageTk, Image
from keras.models import load_model
import numpy as np
import math
import PIL.ImageOps
from main import infer
from Model import Model, DecoderType


filename = ""
profile = ""

fnCharList = 'C:/Users/Ian/PycharmProjects/Handwriting/model/charList.txt'
decoderType = DecoderType.BeamSearch
model = Model(open(fnCharList).read(), decoderType, mustRestore=True)

window = Tk()
window.title("Handwriting Imitation and Recognition")
window.geometry('650x450')

profileNames = []
img_width = 28 # width of new image
img_height = 28 # heigth of new image
dim = (img_width, img_height)

def convert_values(gen_img): # converts values from -1 to 1 to 0 to 255 (grayscale)
    return (((gen_img - np.amin(gen_img)) * (255 - 0)) / (np.amax(gen_img) - np.amin(gen_img)))

def generate_image(letter):
    if profile == "ian":
        generator = load_model('C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/Ian/ian_gen_model_%s.h5' % letter)
    elif profile == "gale":
        generator = load_model('C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/Gale/gale_gen_model_%s.h5' % letter)
    elif profile == "johnston":
        generator = load_model('C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/Johnston/johnston_gen_model_%s.h5' % letter)
    noise = np.random.normal(0, 1, (1, 100))
    gen_img = generator.predict(noise) # generate image using noise vector
    del generator
    gen_img = convert_values(gen_img)
    gen_img = Image.fromarray(gen_img[0, :, :, 0]) # only second and third axis are the pixel values
    return gen_img.resize(dim, Image.ANTIALIAS)  # resize to 28x28 image

def generate_sentence(text):
    if len(text) >= 18:
        new_im = Image.new('L', (18 * img_width, math.ceil((len(text)-1) / 18) * img_height))
        new_im = PIL.ImageOps.invert(new_im)
    else:
        new_im = Image.new('L', ((len(text)-1) * img_width, math.ceil((len(text)-1) / 18) * img_height))
        new_im = PIL.ImageOps.invert(new_im)
    x_offset = 0
    y_offset = 0
    for k in range(len(text)-1):
        if text[k] == " ":
            x_offset += img_width
        else:
            gen_img = generate_image(text[k])
            new_im.paste(gen_img, (x_offset, y_offset))
            del gen_img
            x_offset += img_width
        if ((k % 17) == 0) and (k != 0):
            x_offset = 0
            y_offset += img_height
    return new_im, new_im.size

def recognition_button(): # when the "Recognition" button is pressed
    homeButton.place_configure(x=575, y=400)
    recognitionButton.place_forget()
    imitationButton.place_forget()
    fileButton.place_configure(x=225, y=225)

def imitation_button(): # when the "Imitation" button is pressed
    homeButton.place_configure(x=575, y=400)
    recognitionButton.place_forget()
    imitationButton.place_forget()
    profilesLabel.place_configure(x=0, y=0)
    ianButton.place_configure(x=75, y=50)
    galeButton.place_configure(x=75, y=150)
    johnstonButton.place_configure(x=75, y=250)

def getFile(): # open file directory to select image to recognize handwriting in
    global filename
    filename = filedialog.askopenfilename(initialdir='C:/Users/ianfe/PycharmProjects/Handwriting/', title="Select a file",
                                          filetype=(("png", "*.png",), ("All files", ""))) # open file directory

    if filename.endswith(".png"):
        img = Image.open(filename)
        img = img.resize((504, 308), Image.ANTIALIAS)  # resize to 504x308 image to fit the canvas
        window.recognitionImage = recognitionImage = ImageTk.PhotoImage(image=img, size=img.size)
        recognitionImageCanvas.place_configure(x=85, y=50)
        recognitionImageCanvas.create_image(255, 150, image=recognitionImage) # place selected image on recognitionImageCanvas
        afterRecognitionImageCanvas.create_image(255, 150, image=recognitionImage)
        recognizeButton.place_configure(x=260, y=400)
        fileButton.place_forget()

def createNewProfile(): # creates an entry box to enter a new profile name
    profileNameLabel.place_configure(x=185, y=400)
    profileEntry.place_configure(x=265, y=400)

def profile_entry_button():
    global profileNames
    new_profile = profileEntry.get() # get string the user typed
    if len(new_profile) != 0: # check if entry contains text
        profileNames.append(new_profile) # append entered name to end of profileNames list
        profileEntry.delete(0, END) # delete the text in the entry box
    profileEntry.place_forget()
    profileNameLabel.place_forget()

def home_button(): # hide everything but imitation and recognition button when "home" button is pressed
    recognitionButton.place_configure(x=150, y=50)
    imitationButton.place_configure(x=150, y=250)
    profilesLabel.place_forget()
    profileEntry.place_forget()
    ianButton.place_forget()
    galeButton.place_forget()
    johnstonButton.place_forget()
    profileNameLabel.place_forget()
    fileButton.place_forget()
    profileEntry.delete(0, END)
    imitationTextEntry.place_forget()
    homeButton.place_forget()
    createPictureButton.place_forget()
    profilesLabel.config(text='Profiles') # change label at top of screen back to "Profiles"
    createPictureButton.place_forget()
    tryAgainSameProfile.place_forget()
    tryAgainDifferentProfile.place_forget()
    insertImageCanvas.place_forget()
    recognitionImageCanvas.place_forget()
    recognizeButton.place_forget()
    tryAgainRecognize.place_forget()
    rec_label.place_forget()
    afterRecognitionImageCanvas.place_forget()

def ian_button(): # when "Ian" button is pressed on the profiles screen
    global profile
    profile = "ian"
    profilesLabel.config(text='Ian') # change label at top of screen back to "Ian"
    ianButton.place_forget()
    galeButton.place_forget()
    johnstonButton.place_forget()
    profileNameLabel.place_forget()
    imitationTextEntry.place_configure(x=25, y=25)
    createPictureButton.place_configure(x=280, y=400)
    profileEntry.place_forget()

def gale_button(): # when "Ian" button is pressed on the profiles screen
    global profile
    profile = "gale"
    profilesLabel.config(text='Gale') # change label at top of screen back to "Gale"
    ianButton.place_forget()
    galeButton.place_forget()
    johnstonButton.place_forget()
    profileNameLabel.place_forget()
    imitationTextEntry.place_configure(x=25, y=25)
    createPictureButton.place_configure(x=280, y=400)
    profileEntry.place_forget()

def johnston_button(): # when "Ian" button is pressed on the profiles screen
    global profile
    profile = "johnston"
    profilesLabel.config(text='Johnston') # change label at top of screen back to "Johnston"
    ianButton.place_forget()
    galeButton.place_forget()
    johnstonButton.place_forget()
    profileNameLabel.place_forget()
    imitationTextEntry.place_configure(x=25, y=25)
    createPictureButton.place_configure(x=280, y=400)
    profileEntry.place_forget()

def create_picture_button(): # create image of the sentence typed
    text = imitationTextEntry.get("1.0", END)
    if text != "\n": # check if any text was entered
        if len(text) < 128:
            new_im, img_size = generate_sentence(text)
            window.new_im_tk = new_im_tk = ImageTk.PhotoImage(image=new_im, size=img_size)
            imitationTextEntry.place_forget()
            createPictureButton.place_forget()
            tryAgainSameProfile.place_configure(x=50, y=385)
            tryAgainDifferentProfile.place_configure(x=150, y=385)
            insertImageCanvas.place_configure(x=85, y=50)
            insertImageCanvas.create_image(255, 150, image=new_im_tk)
            imitationTextEntry.delete("1.0", END)
        else:
            messagebox.showerror("Error", "Due to limitations with the image size, "
                                          "the sentence must be less than 128 characters. Please try again.")
    else: # display error message if no text was entered
        messagebox.showerror("Error", "No text detected. Please try again.")

def same_profile_button(): # repeat imitation with same profile
    imitationTextEntry.place_configure(x=25, y=25)
    createPictureButton.place_configure(x=280, y=400)
    insertImageCanvas.place_forget()
    tryAgainSameProfile.place_forget()
    tryAgainDifferentProfile.place_forget()

def different_profile_button(): # go back to profiles screen
    profilesLabel.config(text='Profiles') # change label at top of screen back to "Profiles"
    profilesLabel.place_configure(x=0, y=0)
    ianButton.place_configure(x=75, y=50)
    galeButton.place_configure(x=75, y=150)
    johnstonButton.place_configure(x=75, y=250)
    insertImageCanvas.place_forget()
    tryAgainSameProfile.place_forget()
    tryAgainDifferentProfile.place_forget()

def recognize_button(): # submit photo for handwriting to be recognized in
    rec_word = infer(model, filename)
    rec_label.configure(text="Recognized text is: "+rec_word, font=("Times",32))
    if len(rec_word) > 10:
        rec_label.place_configure(x=60, y=325)
    elif len(rec_word) < 5:
        rec_label.place_configure(x=135, y=325)
    elif len(rec_word) < 5:
        rec_label.place_configure(x=115, y=325)
    else:
        rec_label.place_configure(x=90, y=325)
    afterRecognitionImageCanvas.place_configure(x=75, y=20)
    recognizeButton.place_forget()
    tryAgainRecognize.place_configure(x=300, y=400)
    recognitionImageCanvas.place_forget()

def try_again_recognize(): # repeat recognition
    fileButton.place_configure(x=225, y=225)
    rec_label.place_forget()
    tryAgainRecognize.place_forget()
    afterRecognitionImageCanvas.place_forget()

def resize_image(picture): # resize image to fit recognitionImageCanvas
    img = Image.open(picture)
    return img.resize((500, 300), Image.ANTIALIAS)

# all buttons
recognitionButton = Button(window, width=50, height=10, text="Recognition", relief="raised", command=recognition_button)
imitationButton = Button(window, width=50, height=10, text="Imitation", relief="raised", command=imitation_button)
fileButton = Button(window, text="Choose png to recognize handwriting in", command=getFile)
ianButton = Button(window, text="Ian", height=5, width=20, background="lime green", activebackground="lime green", command=ian_button)
galeButton = Button(window, text="Gale", height=5, width=20, background="cyan", activebackground="cyan", command=gale_button)
johnstonButton = Button(window, text="Johnston", height=5, width=20, background="orange2", activebackground="orange2", command=johnston_button)
homeButton = Button(window, text="Home", command=home_button)
createPictureButton = Button(window, text="Create picture of text", command=create_picture_button)
tryAgainSameProfile = Button(window, text="Try again with\nsame profile", command=same_profile_button)
tryAgainDifferentProfile = Button(window, text="Try again with\ndifferent profile", command=different_profile_button)
recognizeButton = Button(window, text="Recognize text in image", command=recognize_button)
tryAgainRecognize = Button(window, text="Try again", command=try_again_recognize)

# all text entries
profileEntry = Entry(window, width=20)
imitationTextEntry = Text(window, height=22, width=75)

# all labels
profilesLabel = Label(window, text="Profiles", width=93, bd=1, relief=SUNKEN)
profileNameLabel = Label(window, text="Profile Name: ")

# all canvases
insertImageCanvas = Canvas(window, height=308, width=504)
recognitionImageCanvas = Canvas(window, height=308, width=504)
afterRecognitionImageCanvas = Canvas(window, height=308, width=504)

# place "recognition" and "imitation" button on startup
recognitionButton.place_configure(x=150, y=50)
imitationButton.place_configure(x=150, y=250)

rec_label = Label(window)


window.mainloop()
