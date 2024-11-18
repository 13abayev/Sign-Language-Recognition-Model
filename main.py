import warnings
import json
import pickle
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

class GestureRecognition:
    def __init__(self):
        print("Starting Gesture Recognition")
        
        # Load class labels
        with open("datas/dataset/classes.json", "r") as f:
            self.classes = json.load(f)
        
        # Load pre-trained model
        self.model = load_model("models/transformer1.keras")
        
        # Initialize MediaPipe modules
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_detection
        self.hand_detector = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.face_detector = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        
        # Video capture 
        self.cap = cv2.VideoCapture(0)
        self.frames = []
        self.duration = 0
        self.text = ""
        print("Initialized detectors")

    def loadData(self, name, prefix="datas/demo dataset/", file_format="pkl"):
        with open(f"{prefix}{name}.{file_format}", "rb") as f:
            return pickle.load(f)

    def padSample(self, sample, target_length=40):
        if sample.shape[0] < target_length:
            difference = target_length - sample.shape[0]
            padding = np.zeros((difference, sample.shape[1]))
            return np.vstack((padding, sample))
        return sample

    def shiftSample(self, sample):
        for i, frame in enumerate(sample):
            frame = frame.reshape(-1, 2)
            x_values = frame[:, 0]
            y_values = frame[:, 1]
            if np.max(x_values) != 0 and np.max(y_values) != 0:
                min_x = np.min(x_values[x_values > 0])
                min_y = np.min(y_values[y_values > 0])
                x_values = np.clip(x_values - min_x, 0, None)
                y_values = np.clip(y_values - min_y, 0, None)
            sample[i] = np.column_stack((x_values, y_values)).flatten()
        return sample

    def flatten(self, frame):
        hands, head = frame
        hand1_coords = [coord for pair in hands[0] for coord in pair]
        hand2_coords = [coord for pair in hands[1] for coord in pair]
        head_coords = list(head)
        return np.array(hand1_coords + hand2_coords + head_coords)

    def processSamples(self, X):
        for i, sample in enumerate(X):
            sample = np.array([self.flatten(frame) for frame in sample])
            sample = self.shiftSample(sample)
            sample = self.padSample(sample)
            X[i] = sample
        
        X = np.array(X)
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        scaler = self.loadData("scaler")
        X_scaled = scaler.transform(X_reshaped)
        return X_scaled.reshape(X.shape)

    def start(self):
        print("Starting action detection")
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                self.duration -= 1
                
                if not ret:
                    print("Video streaming stopped.")
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_detected, head = self.detectFace(frame, frame_rgb)

                hand_detected, hands = self.detectHands(frame, frame_rgb)

                self.frames.append((hands, head))
                if len(self.frames) > 40:
                    self.frames.pop(0)
                
                if len(self.frames) == 40 and face_detected and hand_detected:
                    self.predictGesture()

                self.displayResult(frame)

                cv2.imshow('Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error: {e}")

    def detectFace(self, frame, frame_rgb):
        face_detected = True
        head = (0, 0)
        face_results = self.face_detector.process(frame_rgb)
        
        if face_results.detections:
            nose = face_results.detections[0].location_data.relative_keypoints[2]
            head = (int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0]))
            self.mp_drawing.draw_detection(frame, face_results.detections[0])
        else:
            face_detected = False
            print("Warning: Could not detect head.", flush=True, end="\r")
        
        return face_detected, head

    def detectHands(self, frame, frame_rgb):
        hand_detected = True
        hands = [[[0, 0] for _ in range(21)] for _ in range(2)]
        hand_results = self.hand_detector.process(frame_rgb)
        
        if hand_results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                hand = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in hand_landmarks.landmark]
                if i < len(hands):
                    hands[i] = hand
        else:
            hand_detected = False
            print("Warning: Could not detect hands.", flush=True, end="\r")
        
        return hand_detected, hands

    def predictGesture(self):
        X = [self.frames]
        X = self.processSamples(X)
        
        probabilities = self.model.predict(X)[0]
        prediction = np.argmax(probabilities)
        confidence = np.max(probabilities) * 100
        
        if confidence > 85:
            self.duration = 60
            self.frames.clear()
            self.text = self.classes[prediction - 1]

    def displayResult(self, frame):
        font = cv2.FONT_HERSHEY_COMPLEX
        font_size = 1.5
        font_thickness = 2
        text_color = (255, 255, 255)
        background_color = (0, 0, 0)
        shadow_color = (50, 50, 50)
        position = (10, 50)

        if self.duration > 0:
            (text_width, text_height), baseline = cv2.getTextSize(self.text, font, font_size, font_thickness)
            cv2.putText(frame, self.text, (position[0] + 2, position[1] + 2), font, font_size, shadow_color, font_thickness + 2)
            cv2.rectangle(frame, 
                          (position[0] - 5, position[1] - text_height - 5),  # top-left corner
                          (position[0] + text_width + 5, position[1] + baseline + 5),  # bottom-right corner
                          background_color, 
                          thickness=-1)
            cv2.putText(frame, self.text, position, font, font_size, text_color, font_thickness)

# Run the gesture recognition
if __name__ == "__main__":
    recognizer = GestureRecognition()
    recognizer.start()
