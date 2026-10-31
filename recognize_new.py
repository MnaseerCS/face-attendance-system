import cv2
import os
import csv
import random
import time
import numpy as np
from datetime import datetime

detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx", "", (320, 320), 0.9, 0.3, 5000
)
recognizer = cv2.FaceRecognizerSF.create(
    "face_recognition_sface_2021dec.onnx", ""
)

THRESHOLD = 0.65          # higher score = better match
FRAMES_NEEDED = 5         # same person seen this many frames in a row
TURN_LIMIT = 0.25         # how far the nose must move to count as a turn
STRAIGHT_LIMIT = 0.12     # how centered the face must be to count as "straight"
CHALLENGE_SECONDS = 6     # time allowed for each step
DIRECTION_SIGN = 1        # change to -1 if left and right feel reversed
LOG_FILE = "attendance.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Date", "Time"])


def already_marked_today(name):
    today = datetime.now().strftime("%Y-%m-%d")
    with open(LOG_FILE, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0] == name and row[1] == today:
                return True
    return False


def mark(name):
    now = datetime.now()
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow(
            [name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")]
        )
    print("Marked present:", name)


# Load everyone who was enrolled
people = {}
for file in os.listdir("embeddings"):
    if file.endswith(".npy"):
        person = file[:-4]
        saved = np.load(os.path.join("embeddings", file))
        people[person] = [f.reshape(1, -1).astype(np.float32) for f in saved]

print("Loaded:", list(people.keys()))


def identify(frame, face):
    aligned = recognizer.alignCrop(frame, face)
    feature = recognizer.feature(aligned)
    best_name, best_score = "Unknown", -1.0
    for person, saved_features in people.items():
        scores = [
            recognizer.match(feature, f, cv2.FaceRecognizerSF_FR_COSINE)
            for f in saved_features
        ]
        score = float(np.mean(sorted(scores)[-3:]))
        if score > best_score:
            best_name, best_score = person, score
    return best_name, best_score


def head_turn(face):
    """0 = looking straight. Positive = nose moved to the right side of the
    image (the person's left). Negative = person's right."""
    eye_a, eye_b, nose = face[4], face[6], face[8]
    eye_mid = (eye_a + eye_b) / 2
    eye_dist = abs(eye_b - eye_a)
    if eye_dist == 0:
        return 0.0
    return DIRECTION_SIGN * (nose - eye_mid) / eye_dist


camera = cv2.VideoCapture(0)

state = "WAIT"            # WAIT -> TURN -> STRAIGHT -> back to WAIT
candidate, streak = None, 0
target, deadline = None, 0
message, msg_until = "", 0

while True:
    ok, frame = camera.read()
    if not ok:
        break

    now = time.time()
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    face = None
    if faces is not None:
        face = faces[np.argmax(faces[:, 2])]  # the biggest face = closest person

    text = "Look at the camera"
    color = (255, 255, 255)

    if state == "WAIT":
        if face is not None:
            name, score = identify(frame, face)
            if score >= THRESHOLD:
                text = f"{name} ({score:.2f})"
                color = (0, 255, 0)
                if name == candidate:
                    streak += 1
                else:
                    candidate, streak = name, 1

                if streak >= FRAMES_NEEDED and now > msg_until:
                    if already_marked_today(name):
                        text = f"{name} - already marked today"
                    elif abs(head_turn(face)) < STRAIGHT_LIMIT:
                        state = "TURN"
                        target = random.choice(["LEFT", "RIGHT"])
                        deadline = now + CHALLENGE_SECONDS
            else:
                text = f"Unknown ({score:.2f})"
                color = (0, 0, 255)
                candidate, streak = None, 0
        else:
            candidate, streak = None, 0

    elif state == "TURN":
        text = f"{candidate}: turn your head to YOUR {target}"
        color = (0, 255, 255)
        if now > deadline:
            state, candidate, streak = "WAIT", None, 0
            message, msg_until = "Liveness check failed - try again", now + 3
        elif face is not None:
            t = head_turn(face)
            if (target == "LEFT" and t > TURN_LIMIT) or (
                target == "RIGHT" and t < -TURN_LIMIT
            ):
                state = "STRAIGHT"
                deadline = now + CHALLENGE_SECONDS

    elif state == "STRAIGHT":
        text = f"{candidate}: now look straight at the camera"
        color = (0, 255, 255)
        if now > deadline:
            state, candidate, streak = "WAIT", None, 0
            message, msg_until = "Liveness check failed - try again", now + 3
        elif face is not None and abs(head_turn(face)) < STRAIGHT_LIMIT:
            name, score = identify(frame, face)
            if name == candidate and score >= THRESHOLD:
                mark(candidate)
                message, msg_until = f"{candidate} marked present", now + 3
                state, candidate, streak = "WAIT", None, 0

    if now < msg_until:
        text = message
        color = (0, 255, 0) if "marked" in message else (0, 0, 255)

    if face is not None:
        x, y, bw, bh = face[:4].astype(int)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

    cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Attendance - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()