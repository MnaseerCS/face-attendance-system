import cv2
import os
import numpy as np

name = input("Enter the person's name: ").strip()
os.makedirs("embeddings", exist_ok=True)
save_path = os.path.join("embeddings", name + ".npy")

# If this person was saved before, keep the old ones and add to them
if os.path.exists(save_path):
    features = list(np.load(save_path))
else:
    features = []

detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx", "", (320, 320), 0.9, 0.3, 5000
)
recognizer = cv2.FaceRecognizerSF.create(
    "face_recognition_sface_2021dec.onnx", ""
)

camera = cv2.VideoCapture(0)
print("Saved so far for", name, ":", len(features))

while True:
    ok, frame = camera.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, bw, bh = face[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    cv2.imshow("Enroll - press s to save, q to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("s"), ord("S")):
        if faces is None:
            print("No face seen - move closer or fix the light")
        else:
            aligned = recognizer.alignCrop(frame, faces[0])
            feature = recognizer.feature(aligned)
            features.append(feature[0])
            np.save(save_path, np.array(features))
            print("Saved", len(features))
    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()