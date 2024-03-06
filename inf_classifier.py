import pickle
import cv2
import mediapipe as mp
import numpy as np


# reading the model from model.p file and storing it in a variable model
model_dict = pickle.load(open('model.p', 'rb'))

# storing the model in a variable model
model = model_dict['model']

# capturing the video from the webcam
cap = cv2.VideoCapture(0)

# mp.solutions.hands is the module for detecting hands in the video
mp_hands = mp.solutions.hands

# mp.solutions.drawing_utils is the module for drawing the detected hands in the video
mp_hand_drawing = mp.solutions.drawing_utils

# mp.solutions.drawing_styles is the module for styling the detected hands in the video
mp_hand_drawing_styles = mp.solutions.drawing_styles

# creating an object called hand from the class mp_hands.Hands
hand = mp_hands.Hands(static_image_mode=True,
                      max_num_hands=1, min_detection_confidence=0.3)

# Assigning the labels to the numbers for ease of use later
label_dict = {0: 'A', 1: 'B', 2: 'C'}


# looping through the video frame by frame
while True:

    # creating an empty list to store the data for each image
    data_auxilary = []

    # creating empty lists to store the x and y coordinates of the landmarks
    x_ = []
    y_ = []

    # reading the frame from the video and storing it in a variable frame
    ret, frame = cap.read()

    # converting the frame from BGR to RGB using cv2.cvtColor and storing it in a variable frame_rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # using hand.process to detect the hand landmarks in the frame and storing the result in a variable result
    result = hand.process(frame_rgb)

    # checking if the result has multi_hand_landmarks
    if result.multi_hand_landmarks:

        # looping through the hand landmarks in the result and drawing the landmarks and connections on the frame
        for hand_landmarks in result.multi_hand_landmarks:

            # drawing the landmarks and connections on the frame
            mp_hand_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_hand_drawing_styles.get_default_hand_landmarks_style(
            ), mp_hand_drawing_styles.get_default_hand_connections_style())

        # looping through the hand landmarks in the result and storing the x and y coordinates of the landmarks in the lists x_ and y_
        for hand_landmarks in result.multi_hand_landmarks:

            # looping through the landmarks in the hand_landmarks
            for i in range(len(hand_landmarks.landmark)):

                # storing the x and y coordinates of the landmarks in the lists x_ and y_
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                data_auxilary.append(x)
                data_auxilary.append(y)

                x_.append(x)
                y_.append(y)

        # drawing a rectangle around the hand in the frame using the x and y coordinates of the landmarks
        x1 = int(min(x_) * frame.shape[1])
        x2 = int(max(x_) * frame.shape[1])
        y1 = int(min(y_) * frame.shape[0])
        y2 = int(max(y_) * frame.shape[0])

        # predicting the hand sign using the model and storing the result in a variable prediction
        prediction = model.predict([np.array(data_auxilary)])

        # storing the predicted hand sign in a variable pred_char
        pred_char = label_dict[int(prediction[0])]

        # drawing the rectangle and the predicted hand sign on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

        # putting the predicted hand sign on the frame
        cv2.putText(frame, pred_char, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # showing the frame
    cv2.imshow('frame', frame)

    # checking if the user has pressed the escape key
    cv2.waitKey(25)

# releasing the video capture object
cap.release()

# destroying all the windows
cv2.destroyAllWindows()
