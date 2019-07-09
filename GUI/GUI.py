from tkinter import *
from tkinter import filedialog
#from tkinter.ttk import *
from tkinter import messagebox
from PIL import ImageTk, Image
from keras.models import load_model
import numpy as np
import math
#from tkinter import messagebox

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
    generator = load_model('C:/Users/Ian/PycharmProjects/Handwriting/Saved-Generator-Models/gen_model_%s.h5' % letter)
    noise = np.random.normal(0, 1, (1, 100))
    gen_img = generator.predict(noise) # generate image using noise vector
    del generator
    gen_img = convert_values(gen_img)
    gen_img = Image.fromarray(gen_img[0, :, :, 0]) # only second and third axis are the pixel values
    return gen_img.resize(dim, Image.ANTIALIAS)  # resize to 28x28 image

def generate_sentence(text):
    if len(text) >= 18:
        new_im = Image.new('RGB', (18 * img_width, math.ceil(len(text) / 18) * img_height))
    else:
        new_im = Image.new('RGB', ((len(text)-1) * img_width, math.ceil(len(text) / 18) * img_height))
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
    createProfileButton.place_configure(x=275, y=400)
    ianButton.place_configure(x=75, y=50)
    ianDeleteButton.place_configure(x=425, y=50)

def getFile(): # open file directory to select image to recognize handwriting in
    filename = filedialog.askopenfilename(initialdir='C:/Users/ianfe/PycharmProjects/Handwriting/', title="Select a file",
                                          filetype=(("png", "*.png",), ("All files", ""))) # open file directory

    if filename.endswith(".png"):
        img = Image.open(filename)
        img = img.resize((504, 308), Image.ANTIALIAS)  # resize to 28x28 image
        window.recognitionImage = recognitionImage = ImageTk.PhotoImage(image=img, size=img.size)
        recognitionImageCanvas.place_configure(x=85, y=50)
        recognitionImageCanvas.create_image(255, 150, image=recognitionImage) # place selected image on recognitionImageCanvas
        recognizeButton.place_configure(x=260, y=400)
        fileButton.place_forget()

def createNewProfile(): # creates an entry box to enter a new profile name
    createProfileButton.place_forget()
    profileNameLabel.place_configure(x=185, y=400)
    profileEntry.place_configure(x=265, y=400)
    makeProfileButton.place_configure(x=395, y=395)

def profile_entry_button():
    global profileNames
    new_profile = profileEntry.get() # get string the user typed
    if len(new_profile) != 0: # check if entry contains text
        profileNames.append(new_profile) # append entered name to end of profileNames list
        profileEntry.delete(0, END) # delete the text in the entry box
    makeProfileButton.place_forget()
    profileEntry.place_forget()
    profileNameLabel.place_forget()
    createProfileButton.place_configure(x=275, y=400)


def home_button(): # hide everything but imitation and recognition button when "home" button is pressed
    recognitionButton.place_configure(x=150, y=50)
    imitationButton.place_configure(x=150, y=250)
    profilesLabel.place_forget()
    profileEntry.place_forget()
    createProfileButton.place_forget()
    makeProfileButton.place_forget()
    ianButton.place_forget()
    profileNameLabel.place_forget()
    fileButton.place_forget()
    profileEntry.delete(0, END)
    imitationTextEntry.place_forget()
    homeButton.place_forget()
    createPictureButton.place_forget()
    ianDeleteButton.place_forget()
    profilesLabel.config(text='Profiles') # change label at top of screen back to "Profiles"
    createPictureButton.place_forget()
    tryAgainSameProfile.place_forget()
    tryAgainDifferentProfile.place_forget()
    insertImageCanvas.place_forget()
    recognitionImageCanvas.place_forget()
    recognizeButton.place_forget()
    tryAgainRecognize.place_forget()
    textHereImageCanvas.place_forget()

def ian_button(): # when "Ian" button is pressed on the profiles screen
    profilesLabel.config(text='Ian') # change label at top of screen back to "Ian"
    createProfileButton.place_forget()
    ianButton.place_forget()
    profileNameLabel.place_forget()
    ianDeleteButton.place_forget()
    imitationTextEntry.place_configure(x=25, y=25)
    createPictureButton.place_configure(x=280, y=400)
    profileEntry.place_forget()
    makeProfileButton.place_forget()

def create_picture_button(): # create image of the sentence typed
    text = imitationTextEntry.get("1.0", END)
    if text != "\n": # check if any text was entered
        new_im, img_size = generate_sentence(text)
        window.new_im_tk = new_im_tk = ImageTk.PhotoImage(image=new_im, size=img_size)
        imitationTextEntry.place_forget()
        createPictureButton.place_forget()
        tryAgainSameProfile.place_configure(x=50, y=385)
        tryAgainDifferentProfile.place_configure(x=150, y=385)
        insertImageCanvas.place_configure(x=85, y=50)
        insertImageCanvas.create_image(255, 150, image=new_im_tk)
        imitationTextEntry.delete("1.0", END)
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
    createProfileButton.place_configure(x=275, y=400)
    ianButton.place_configure(x=75, y=50)
    ianDeleteButton.place_configure(x=425, y=50)
    insertImageCanvas.place_forget()
    tryAgainSameProfile.place_forget()
    tryAgainDifferentProfile.place_forget()

def recognize_button(): # submit photo for handwriting to be recognized in
    textHereImageCanvas.place_configure(x=180, y=200)
    # textHereImageCanvas.create_image(175, 20, image=textHereImage) # placeholder image. will be removed in the future
    recognizeButton.place_forget()
    tryAgainRecognize.place_configure(x=300, y=400)
    recognitionImageCanvas.place_forget()

def try_again_recognize(): # repeat recognition
    fileButton.place_configure(x=225, y=225)
    textHereImageCanvas.place_forget()
    tryAgainRecognize.place_forget()

def resize_image(picture): # resize image to fit recognitionImageCanvas
    img = Image.open(picture)
    return img.resize((500, 300), Image.ANTIALIAS)

# all buttons
recognitionButton = Button(window, width=50, height=10, text="Recognition", relief="raised", command=recognition_button)
imitationButton = Button(window, width=50, height=10, text="Imitation", relief="raised", command=imitation_button)
fileButton = Button(window, text="Choose png to recognize handwriting in", command=getFile)
createProfileButton = Button(window, text="Create new profile", command=createNewProfile)
makeProfileButton = Button(window, text="Make profile", command=profile_entry_button)
ianButton = Button(window, text="Ian", height=5, width=20, background="cyan", activebackground="cyan", command=ian_button)
ianDeleteButton = Button(window, text="Delete", height=5, width=20, background="red", activebackground="red")
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
profileNameLabel = Label(window, text="Profile Name:")

# all canvases
insertImageCanvas = Canvas(window, height=308, width=504)
textHereImageCanvas = Canvas(window, height=59, width=362)
recognitionImageCanvas = Canvas(window, height=308, width=504)

# place "recognition" and "imitation" button on startup
recognitionButton.place_configure(x=150, y=50)
imitationButton.place_configure(x=150, y=250)

window.mainloop()
