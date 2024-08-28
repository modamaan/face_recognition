from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import threading
import pygame

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# List of names and their corresponding image file paths
face_data = [
    {"name": "Mohamed amaan", "files": ["./Amaan/amaan1.jpg", "./Amaan/amaan1.jpg"]},
    {"name": "Neymar jr", "files": ["./Neymar/neymar.jpg"]},
    {"name": "NIzam", "files": ["./Nizam/nizamai.jpg"]},
    {"name": "Adhithyan", "files": ["./Adhithyan/adhithyan.jpeg"]},
    {"name": "Avin M.M", "files": ["./Avin/avin.jpg"]},
    {"name": "Joel", "files": ["./Joyal/joyal.jpg"]}
]

# Initialize lists for known face encodings and their names
known_face_encodings = []
known_face_names = []

# Load images and create face encodings
for person in face_data:
    for file_path in person["files"]:
        # Load the image
        image = face_recognition.load_image_file(file_path)
        # Encode the face
        face_encoding = face_recognition.face_encodings(image)[0]
        # Append the encoding and the name to the lists
        known_face_encodings.append(face_encoding)
        known_face_names.append(person["name"])

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
audio_playing = False  # To ensure the audio plays only once per detection
play_count = 0
max_play_count = 3  # Maximum number of times to play the message
welcome_displayed = False  # To track if the welcome message has been displayed

def play_welcome_message():
    global audio_playing, play_count
    if not audio_playing and play_count < max_play_count:
        audio_playing = True
        try:
            pygame.mixer.music.load('./templates/welcome2.mp3')
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():  # Wait for the music to finish playing
                pass
            play_count += 1
            print(f"Audio played successfully. Played {play_count} times.")
        except Exception as e:
            print(f"Error playing sound: {e}")
        finally:
            audio_playing = False

def gen_frames():
    global face_locations, face_encodings, face_names, welcome_displayed
    while True:
        success, frame = camera.read()  # Read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

                # Check if "Mohamed amaan" is recognized and play audio and display welcome message
                if name == "Mohamed amaan":
                    if not welcome_displayed:
                        threading.Thread(target=play_welcome_message).start()
                        welcome_displayed = True
                else:
                    welcome_displayed = False

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

                # If Mohamed amaan is detected, display a welcome message
                if name == "Mohamed amaan":
                    cv2.putText(frame, "Welcome Mohamed Amaan", (left - 17, top - 10), font, 0.75, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)



# from flask import Flask, render_template, Response
# import cv2
# import face_recognition
# import numpy as np
# app=Flask(__name__)
# camera = cv2.VideoCapture(0)
# # Load a sample picture and learn how to recognize it.
# amaan_image = face_recognition.load_image_file("./Amaan/amaan.jpg")
# amaan_face_encoding = face_recognition.face_encodings(amaan_image)[0]

# # Load a second sample picture and learn how to recognize it.
# neymar_image = face_recognition.load_image_file("./Neymar/neymar.jpg")
# neymar_face_encoding = face_recognition.face_encodings(neymar_image)[0]

# nizam_image = face_recognition.load_image_file("./Nizam/nizamai.jpg")
# nizam_face_encoding = face_recognition.face_encodings(nizam_image)[0]

# adhithyan_image = face_recognition.load_image_file("./Adhithyan/adhithyan.jpeg")
# adhithyan_face_encoding = face_recognition.face_encodings(adhithyan_image)[0]

# avin_image = face_recognition.load_image_file("./Avin/avin.jpg")
# avin_face_encoding = face_recognition.face_encodings(avin_image)[0]

# joel_image = face_recognition.load_image_file("./Joyal/joyal.jpg")
# joel_face_encoding = face_recognition.face_encodings(joel_image)[0]



# # Create arrays of known face encodings and their names
# known_face_encodings = [
#     amaan_face_encoding,
#     neymar_face_encoding,
#     nizam_face_encoding,
#     adhithyan_face_encoding,
#     avin_face_encoding,
#     joel_face_encoding
# ]
# known_face_names = [
#     "Mohamed amaan",
#     "Neymar jr",
#     "NIzam",
#     "Adhithyan",
#     "Avin M.M",
#     "Joel"

# ]
# # Initialize some variables
# face_locations = []
# face_encodings = []
# face_names = []
# process_this_frame = True

# def gen_frames():
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             # Resize frame of video to 1/4 size for faster face recognition processing
#             small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#             # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#             # rgb_small_frame = small_frame[:, :, ::-1]
#             rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#             # Only process every other frame of video to save time

#             # Find all the faces and face encodings in the current frame of video
#             face_locations = face_recognition.face_locations(rgb_small_frame)
#             face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
#             face_names = []
#             for face_encoding in face_encodings:
#                 # See if the face is a match for the known face(s)
#                 matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#                 name = "Unknown"
#                 # Or instead, use the known face with the smallest distance to the new face
#                 face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#                 best_match_index = np.argmin(face_distances)
#                 if matches[best_match_index]:
#                     name = known_face_names[best_match_index]

#                 face_names.append(name)


#             # Display the results
#             for (top, right, bottom, left), name in zip(face_locations, face_names):
#                 # Scale back up face locations since the frame we detected in was scaled to 1/4 size
#                 top *= 4
#                 right *= 4
#                 bottom *= 4
#                 left *= 4

#                 # Draw a box around the face
#                 cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#                 # Draw a label with a name below the face
#                 cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#                 font = cv2.FONT_HERSHEY_DUPLEX
#                 cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# if __name__=='__main__':
#     app.run(debug=True)
