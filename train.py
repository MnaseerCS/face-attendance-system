import cv2
import os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()

photos = []
labels = []
names = []

for label, name in enumerate(sorted(os.listdir("faces"))):
    names.append(name)
    folder = os.path.join("faces", name)
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 200))
        img = cv2.equalizeHist(img)
        photos.append(img)
        labels.append(label)

recognizer.train(photos, np.array(labels))
recognizer.save("trainer.yml")

with open("names.txt", "w") as f:
    f.write("\n".join(names))

print("Trained on", len(photos), "photos of", len(names), "people:", names)