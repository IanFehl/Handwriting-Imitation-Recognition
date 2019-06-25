from tkinter import *
from tkinter import filedialog
#from tkinter.ttk import *
from tkinter import messagebox
from PIL import ImageTk, Image
#from tkinter import messagebox

window = Tk()
window.title("Handwriting Imitation and Recognition")
window.geometry('650x450')

profileNames = []
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

    # must pre-load images to display before program runs due to issues with creating a PhotoImage based on the string
    # "filename[43:]"
    if   filename[43:] == "IanPangram1.png":
        recognitionImage = ianPangram1
    elif filename[43:] == "IanPangram2.png":
        recognitionImage = ianPangram2
    elif filename[43:] == "IanPangram3.png":
        recognitionImage = ianPangram3
    elif filename[43:] == "IanPangram4.png":
        recognitionImage = ianPangram4
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
    if imitationTextEntry.get("1.0", END) != "\n": # check if any text was entered
        imitationTextEntry.place_forget()
        createPictureButton.place_forget()
        tryAgainSameProfile.place_configure(x=50, y=385)
        tryAgainDifferentProfile.place_configure(x=150, y=385)
        insertImageCanvas.place_configure(x=200, y=200)
        insertImageCanvas.create_image(150, 20, image=insertImage) # placeholder image. will be removed in the future
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
    textHereImageCanvas.create_image(175, 20, image=textHereImage) # placeholder image. will be removed in the future
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

insertImage = ImageTk.PhotoImage(Image.open("InsertImageHere.png")) # placeholder image. will be removed in the future
textHereImage = ImageTk.PhotoImage(Image.open("TextGoesHere.png")) # placeholder image. will be removed in the future

# Resize all the images to fit recognitionImageCanvas
ianPangram1 = resize_image("IanPangram1.png")
ianPangram2 = resize_image("IanPangram2.png")
ianPangram3 = resize_image("IanPangram3.png")
ianPangram4 = resize_image("IanPangram4.png")

ianPangram1 = ImageTk.PhotoImage(ianPangram1)
ianPangram2 = ImageTk.PhotoImage(ianPangram2)
ianPangram3 = ImageTk.PhotoImage(ianPangram3)
ianPangram4 = ImageTk.PhotoImage(ianPangram4)

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
insertImageCanvas = Canvas(window, height=78, width=319)
textHereImageCanvas = Canvas(window, height=59, width=362)
recognitionImageCanvas = Canvas(window, height=300, width=500)

# place "recognition" and "imitation" button on startup
recognitionButton.place_configure(x=150, y=50)
imitationButton.place_configure(x=150, y=250)

window.mainloop()
