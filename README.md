# Face Attendance System

A face-recognition attendance system with a liveness check. It watches a camera, recognizes enrolled people, makes sure a real person is present (not a photo), and records each person's attendance once per day in a CSV file.

Built as a capstone project at NASCOMSoft Embedded (Bauchi) by Muhammad Nasir Hussaini. This version uses face only. A fingerprint sensor is planned as a second biometric.

## What it does

1. **Finds faces** in live camera video (YuNet face detector).
2. **Recognizes who it is** by turning each face into a list of numbers and comparing it with the enrolled people (SFace face recognizer).
3. **Checks the person is live** with a random head-turn challenge ("turn your head to your left or right"), then asks them to look straight at the camera again. A flat photo cannot follow the instruction.
4. **Records attendance** in `attendance.csv` (name, date, time), once per person per day.

## How well it works

The first version used a classic method (LBPH). In testing, its scores for correct and wrong matches overlapped, so no threshold could separate them. Brothers and sisters of similar age were mixed up, and strangers were given names.

The current version uses OpenCV's SFace model.

| Case | Similarity score |
|---|---|
| Enrolled person, correct name | 0.81 to 0.91 |
| Unknown people | 0.39 to 0.48 |

The gap between the two lets the system use a threshold of 0.65, so unknown people are shown as "Unknown". In testing, a photo of an enrolled person shown on a phone was recognized by the face check alone. With the head-turn liveness check added, it no longer marks attendance.

## Requirements

- Python 3.10 or newer (tested on Python 3.14)
- A webcam
- Python packages:

```
pip install opencv-contrib-python==4.11.0.86 numpy
```

OpenCV 5.x is not supported by these scripts, so keep to the 4.x version above.

### Model files (download once)

The two model files are not stored in this repository because of their size. Download them and place them in the project folder, next to the scripts:

- `face_detection_yunet_2023mar.onnx` from the Hugging Face repository `opencv/face_detection_yunet`
- `face_recognition_sface_2021dec.onnx` from the Hugging Face repository `opencv/face_recognition_sface`

Do not download the files with "int8" or "quantized" in the name.

## How to use

**1. Enroll each person.** Run this once per person:

```
python enroll_new.py
```

Type the person's name, then press `s` about 10 times while the green box is on their face, turning their head slightly between presses. Press `q` to finish. Use the same spelling of the name each time. Only one person should be in front of the camera.

**2. Run the attendance system:**

```
python recognize_new.py
```

- Look at the camera. When you are recognized, your name appears in green.
- Follow the on-screen instruction and turn your head to the side that it asks for.
- Then look straight at the camera again.
- Your attendance is saved to `attendance.csv`.

Press `q` to quit. Close `attendance.csv` in Excel before running the program, because Windows locks an open file.

### Settings you can change

These are at the top of `recognize_new.py`:

| Setting | What it does |
|---|---|
| `THRESHOLD` | Minimum similarity to accept a match. Higher is stricter. |
| `FRAMES_NEEDED` | How many frames in a row the same person must be seen. |
| `TURN_LIMIT` | How far the head must turn to count as a turn. |
| `DIRECTION_SIGN` | Change to `-1` if left and right feel reversed. |
| `CHALLENGE_SECONDS` | Time allowed for each liveness step. |

## Project files

| File | Purpose |
|---|---|
| `enroll_new.py` | Enroll a person (saves face features to `embeddings/`). |
| `recognize_new.py` | Live recognition, liveness check, and attendance log. |
| `Biometric.py`, `train.py`, `recognize.py` | First version using the LBPH method. Kept for comparison. |

## Privacy

This project stores people's faces, so the following are **not** part of the repository (see `.gitignore`) and stay on the local computer:

- `faces/` (enrollment photos from the first version)
- `embeddings/` (saved face features)
- `attendance.csv`
- `trainer.yml`, `names.txt`
- `*.onnx` model files

Only enroll people who know about it and agree. Keep these folders private.

## Limitations

- The head-turn check is a first layer of protection. A determined person could imitate a turn by tilting a phone, and a pre-recorded video is harder, but not impossible, to use.
- Recognition depends on lighting and distance. Stand about an arm's length from the camera. A face that is too close gets cut off and scores low.
- Only one person should be in front of the camera at a time.
- Tested with a small group of enrolled people, not a full class.

## Future work

- Add fingerprint recognition as a second biometric (multimodal).
- Stronger liveness detection (blink detection, an anti-spoofing model, or an infrared/depth camera).
- Run on a Raspberry Pi.
- Store attendance in a database such as SQLite.
