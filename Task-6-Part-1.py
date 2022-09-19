# User1 Face

import cv2
import numpy as np

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x:x + w]
    return cropped_face


# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = 'faces/user/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100:  # 13 is the Enter Key
        break

cap.release()

cv2.destroyAllWindows()
print("Collecting Face Samples Complete for User-1")


# User2 Face

import cv2
import numpy as np

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')


# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x:x + w]
    return cropped_face


# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = 'faces/user2/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100:  # 13 is the Enter Key
        break

cap.release()

cv2.destroyAllWindows()
print("Collecting Face Samples Complete for User-2")

# Model Training


import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Get the training data we previously made
data_path = 'faces/user/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(1)

data_path = 'faces/user2/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

for i, files in enumerate(onlyfiles):
    img_path = data_path + onlyfiles[i]
    images = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(2)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
model = cv2.face.LBPHFaceRecognizer_create()

# Let's train our model
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")

import cv2
import numpy as np
import time
import os

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')


def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Open Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = model.predict(face)
        if results[1] < 500:
            confidence = int(100 * (1 - (results[1]) / 400))
            display_string = str(confidence) + '% Confident it is User'

        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 120, 150), 2)

        if confidence > 90:
            cv2.putText(image, "Vivek, Your face has been recognized!", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.imshow('Face Recognition', image)
            # to send whatsapp message

            # send_mail() # calling the function
            break


        else:
            cv2.putText(image, "I guess, wrong face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Recognition', image)

    except:
        cv2.putText(image, "No Face Found", (220, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Face Recognition', image)
        pass

        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break

cap.release()
cv2.destroyAllWindows()

import smtplib
import imghdr
import getpass
from email.message import EmailMessage

print("\t\t\t\n=========================================================\n")
Sender_Email = input("Enter Sender Mail Id : ")
print("\t\t\t\n=========================================================\n")
Reciever_Email = input("Enter Reciever Mail id : ")
#print("\t\t\t\n=========================================================\n")
Password = getpass.getpass("Enter your password")
print("\t\t\t\n=========================================================\n")

newMessage = EmailMessage()
newMessage['Subject'] = "face Recognition."
newMessage['From'] = Sender_Email
newMessage['To'] = Reciever_Email
newMessage.set_content('*alert* Your face has been recognized')

with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    smtp.login(Sender_Email, Password)
    smtp.send_message(newMessage)

print("\t\t\t\n======================= Mail Successfully Sent ==================================\n")

import pywhatkit as kit

kit.sendwhatmsg_instantly(phone_no="+3530894561814",
                          message="Vivek, Your face has been recognized and sending whatsapp confirmation!")

import os

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')


def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Open Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # "results" comprises of a tuple containing the label and the confidence value
        results = model.predict(face)
        if results[1] < 500:
            confidence = int(100 * (1 - (results[1]) / 400))
            display_string = str(confidence) + '% Confident it is User'

        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 120, 150), 2)

        if confidence > 90:
            cv2.putText(image, "YOUR FACE HAS BEEN RECOGNIZED!!", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
                        2)
            cv2.imshow('Face Recognition', image)

            # Launching an ec2 instance and a 5GB instance volume if second face is recognized:

            # Launching an ec2 instance
            os.system("aws ec2 run-instances  --image-id ami-0b2ca94b5b49e0132 --instance-type t2.micro  --count 1  --subnet-id subnet-11a93a77 --security-group-ids sg-0808efc79035be1b0 --key-name t6key")
            print("An Ec2 Instance is Launched Successfully")

            # For launching 5GB EBS Volume - we are creating and attaching it.
            os.system("aws ec2 create-volume  --availability-zone us-west-1a  --size 5  --volume-type gp2")
            print("An EBS Volume of Size 5 GB has been created")

            break


        else:
            cv2.putText(image, "I dont know, who r u", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Recognition', image)

    except:
        cv2.putText(image, "No Face Found", (220, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Face Recognition', image)
        pass

        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break

cap.release()
cv2.destroyAllWindows()
