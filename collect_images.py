import os
import cv2

# directory to save the images
data_dir = 'images/'

# if the directory doesn't exist then os would automatically create the directory
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# capture the video from default camera
cap = cv2.VideoCapture(0)

# number of classes to collect images for. I have chosen A, B and C as my classes
no_of_classes = 3

# number of samples to collect for each class
no_of_samples = 500

# loop over all the classes to collect images
for i in range(no_of_classes):

    # if the path for the class doesn't exist then os would automatically create the directory
    if not os.path.exists(data_dir + str(i)):
        os.makedirs(data_dir + str(i))

    print('Collecting images for class: ' + str(i))

    # wait for 5 seconds before starting the webcam
    done = False

    # loop over all the frames from the webcam until we reach the desired number of samples
    while True:
        # read the frame from the webcam and store it in the frame and ret variables
        ret, frame = cap.read()

        # adding the text to the frame to guide the user to press s to save the image
        cv2.putText(frame, 'Press s to save image', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # show the frame on the screen
        cv2.imshow('Frame: ' + str(i), frame)

        # wait for the user to press s to save the image
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # counter to keep track of the number of images saved
    counter = 0

    # loop over all the frames from the webcam until we reach the desired number of samples
    while counter < no_of_samples:

        # read the frame from the webcam and store it in the frame and ret variables
        ret, frame = cap.read()

        # adding the text to the frame to guide the user to press s to save the image
        cv2.imshow('Frame: ' + str(i), frame)

        # wait for the user to press s to save the image
        cv2.waitKey(25)

        # save the image in the specified directory with the specified name
        cv2.imwrite(data_dir + str(i) + '/' + str(counter) + '.jpg', frame)

        # increment the counter
        counter += 1

    # print that we are done collecting images for the current class
    print('Done collecting images for class: ' + str(i))

# release the webcam
cap.release()

# close all the windows
cv2.destroyAllWindows()
