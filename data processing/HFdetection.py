
import cv2
import mediapipe as mp

import pickle


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection

# Setup MediaPipe detectors
hand_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def fixFrame(start, end, duration):
    duration -= 40
    if duration < 0:
        return (start, end)
    if duration <= 10:
        start += duration // 2
        end -= (duration - (duration // 2)) 
    else:
        start += 10
        end -= duration - 10
    return (start, end)


def getData(df):
    total = len(df)
    i = 0
    X, Y = [], []
    head = (0, 0)
    for index, row in df.iterrows():
        i += 1
        print(f"Processed word is {row['text']}\t{i}\\{total}")
        prefix = "../datas/videos/"
        video_path = prefix + f'{row["VideoID"]}.mp4'
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            continue
        start_frame, end_frame = fixFrame(row["start"], row["end"], row["duration"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        curr_frame = start_frame
        frames = []
        while cap.isOpened() and curr_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            # frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_detector.process(frame_rgb)
            handResults = hand_detector.process(frame_rgb)
            
            if face_results.detections:
                detection = face_results.detections[0]
                nose = detection.location_data.relative_keypoints[2]
                x = int(nose.x * frame.shape[1])
                y = int(nose.y * frame.shape[0])
                head = (x, y)
            hands = []
            if handResults.multi_hand_landmarks:
                for hand_landmarks in handResults.multi_hand_landmarks:
                    hand = []
                    for landmark in hand_landmarks.landmark:
                        height, width, _ = frame.shape
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        hand.append((x, y))
                    hands.append(hand)
            while len(hands) < 2:
                hands.append([(0, 0) for _ in range(21)])
            curr_frame += 1
            frames.append((hands, head))
        X.append(frames)
        Y.append(row["text"])
        cap.release()
    cv2.destroyAllWindows()
    return X, Y
    


def dumpData(name : str, data, prefix = "../datas/demo dataset/"):
    with open(f"{prefix}{name}.pkl", "wb") as f:
        pickle.dump(data, f)


with open("../datas/demo dataset/trainDF.pkl", "rb") as f:
    trainDF = pickle.load(f)

X_train, y_train = getData(trainDF)
dumpData("X_train", X_train)
dumpData("y_train", y_train)


with open("../datas/demo dataset/testDF.pkl", "rb") as f:
    testDF = pickle.load(f)

X_test, y_test = getData(testDF)
dumpData("X_test", X_test)
dumpData("y_test", y_test)


with open("../datas/demo dataset/valDF.pkl", "rb") as f:
    valDF = pickle.load(f)

X_val, y_val  =getData(valDF)
dumpData("X_val", X_val)
dumpData("y_val", y_val)
