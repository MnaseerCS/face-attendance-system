import cv2
import os

name = input("Enter the person's name: ").strip()
folder = os.path.join("faces", name)
os.makedirs(folder, exist_ok=True)

face_finder = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)
count = len(os.listdir(folder))
print("Photos already saved for", name, ":", count)

while True:
    ok, frame = camera.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_finder.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Enrollment - press s to save, q to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key != 255:  # 255 means no key was pressed
        print("Key heard:", chr(key))

    if key in (ord("s"), ord("S")):
        if len(faces) == 0:
            print("No face seen - move closer or fix the light")
        else:
            x, y, w, h = faces[0]
            face_photo = gray[y:y + h, x:x + w]
            count += 1
            path = os.path.join(folder, f"{count}.jpg")
            saved = cv2.imwrite(path, face_photo)
            print("Saved photo", count, "worked:", saved)
    elif key == ord("q"):
        
        break

camera.release()
cv2.destroyAllWindows()