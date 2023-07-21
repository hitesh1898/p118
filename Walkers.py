import cv2

# Replace 'path_to_cascade.xml' with the actual path to the CascadeClassifier file
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Replace 'path_to_video.mp4' with the actual path to your pre-recorded video file
cap = cv2.VideoCapture('walking.mp4')

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bodies in the frame
    bodies = body_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

    # Display the frame
    cv2.imshow('Body Detection', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(25)==32:
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
