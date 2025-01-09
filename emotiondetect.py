import cv2
from deepface import DeepFace
import numpy as np

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Detect faces in the image using Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Analyze each face found
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Ensure the face area is reasonably large
        if w > 30 and h > 30:  # Adjust sizes as necessary
            face_frame = frame[y:y+h, x:x+w]

            try:
                # Analyze the face for emotions
                results = DeepFace.analyze(face_frame, actions=['emotion'], enforce_detection=False)
                if results and isinstance(results, list) and len(results) > 0:
                    result = results[0]  # Access the first result if it's a list of results
                    dominant_emotion = result['dominant_emotion']  # Directly access the key
                    # Display the dominant emotion
                    cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                else:
                    print("No valid results returned.")

            except KeyError as key_err:
                print("Key error accessing result:", key_err)
            except Exception as e:
                print("Error in emotion detection:", e)
        else:
            print("Face too small for reliable detection")

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam and destroy all windows
video_capture.release()
cv2.destroyAllWindows()
