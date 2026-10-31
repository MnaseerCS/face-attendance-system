import cv2

face_finder = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

with open("names.txt") as f:
    names = f.read().splitlines()

THRESHOLD = 70  # lower distance = better match

camera = cv2.VideoCapture(0)

while True:
    ok, frame = camera.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_finder.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        face = cv2.equalizeHist(face)
        label, distance = recognizer.predict(face)

        if distance < THRESHOLD:
            text = f"{names[label]} ({distance:.0f})"
            color = (0, 255, 0)  # green
        else:
            text = f"Unknown ({distance:.0f})"
            color = (0, 0, 255)  # red

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Recognition - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()