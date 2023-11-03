# To Capture Frame
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("keras_model.h5")
# Load your machine learning model
 # Assuming your model file is named 'keras_model.h5'
labels = open('labels.txt').read().strip().split("\n")

# Attaching Cam indexed as 0, with the application software

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera.")


# Infinite loop
while True:

    # Reading / Requesting a Frame from the Camera 
    status, frame = camera.read()

    # if we were successfully able to read the frame
    if status:

        # Flip the frame
        frame = cv2.flip(frame, 1)

        # Resize the frame
        frame = cv2.resize(frame, (224, 224))  # Adjust dimensions based on your model's input size

        # Expand the dimensions
        frame_expanded = np.expand_dims(frame, axis=0)

        # Normalize the frame
        frame_normalized = frame_expanded / 255.0

        # Get predictions from the model
        predictions = model.predict(frame_normalized)
        max_prediction = np.argmax(predictions)
        gesture_label = labels[max_prediction]

        # Display predictions on the frame
        cv2.putText(frame, f"Prediction: {gesture_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Displaying the frames captured
        cv2.imshow('feed', frame)

        # Waiting for 1ms
        code = cv2.waitKey(1)

        # If space key is pressed, break the loop
        if code == 32:
            break

# Release the camera from the application software
camera.release()

# Close the open window
cv2.destroyAllWindows()
