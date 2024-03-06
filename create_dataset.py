import os
import cv2
import mediapipe as mp
import pickle

# direcrtory of the images to read the data from
data_dir = 'images/'

# remove .DS_Store file from the directory
if '.DS_Store' in os.listdir(data_dir):
    os.remove(os.path.join(data_dir, '.DS_Store'))

# print the folder in the images directory
print(os.listdir(data_dir))

# storing mp.solutions.hands in a variable for ease of use later
# mp.solutions.hands is the module that will be used to detect the hand landmarks in the image
mp_hands = mp.solutions.hands

# storing mp.solutions.drawing_utils in a variable for ease of use later
# mp.solutions.drawing_utils is the module that will be used to draw the landmarks on the image
mp_hand_drawing = mp.solutions.drawing_utils

# storing mp.solutions.drawing_styles in a variable for ease of use later
# mp.solutions.drawing_styles is the module that will be used to draw the landmarks on the image
mp_hand_drawing_styles = mp.solutions.drawing_styles

# storing mp_hands.Hands in a variable for ease of use later
# mp_hands.Hands is the class that will be used to detect the hand landmarks in the image and draw them on the image
hand = mp_hands.Hands(static_image_mode=True,
                      max_num_hands=1, min_detection_confidence=0.3)

# creating empty lists to store the data and labels
data = []
labels = []

# looping through the directories in the data_dir
for dir_ in os.listdir(data_dir):
    # looping through the images in the directory
    for img_path in os.listdir(os.path.join(data_dir, dir_)):

        # creating an empty list to store the data for each image
        data_auxilary = []

        # reading the image using cv2.imread and storing it in a variable img
        img = cv2.imread(os.path.join(data_dir, dir_, img_path))

        # converting the image from BGR to RGB using cv2.cvtColor and storing it in a variable img_rgb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # using hand.process to detect the hand landmarks in the image and storing the result in a variable result
        result = hand.process(img_rgb)

        # checking if the result has any hand landmarks
        if result.multi_hand_landmarks:

            # looping through the hand landmarks in the result
            for hand_landmarks in result.multi_hand_landmarks:
                # mp_hand_drawing.draw_landmarks(img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_hand_drawing_styles.get_default_hand_landmarks_style(), mp_hand_drawing_styles.get_default_hand_connections_style())

                # looping through the landmarks in the hand_landmarks and storing the x and y coordinates of each landmark in the data_auxilary list
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    data_auxilary.append(x)
                    data_auxilary.append(y)

            # resizing the image to 224x224 using cv2.resize and storing it in a variable img_resized
            data.append(data_auxilary)

            # storing the label of the image in the labels list
            labels.append(dir_)

# making a dataset dictionary to store the data and labels
f = open('data.pickle', 'wb')

# dumping the dataset dictionary into the file data.pickle using pickle.dump
pickle.dump({'data': data, 'labels': labels}, f)

# closing the file
f.close()
