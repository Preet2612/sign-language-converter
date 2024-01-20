import cv2
import mediapipe as mp
import pickle
import numpy as np

def load_model(model_path='./model.p'):
    try:
        model_dict = pickle.load(open(model_path, 'rb'))
        return model_dict['model']
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None

model = load_model()

if model is None:
    exit()

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
    19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y', 24: 'Z', 25: 'K'
}

# Assuming the model was trained on a certain number of features, e.g., 84
expected_num_features = 84

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        # trying to get the corners of the rectangle around the hand.
        x1 = max(0, int(min(x_) * W) - 10)
        y1 = max(0, int(min(y_) * H) - 10)
        x2 = min(W, int(max(x_) * W) - 10)
        y2 = min(H, int(max(y_) * H) - 10)

        # Ensure that the number of features matches the expected number
        if len(data_aux) == expected_num_features:
            prediction = model.predict([np.asarray(data_aux)])

            # Check if the predicted value is within the range of labels
            if 0 <= int(prediction[0]) < len(labels_dict):
                predicted_character = labels_dict[int(prediction[0])]

                # Rectangle and text drawing
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)  # White rectangle
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3,
                            cv2.LINE_AA)  # White text
            else:
                print(f"Warning: Predicted value ({prediction[0]}) is not in the range of labels.")

        else:
            print(f"Warning: Number of features ({len(data_aux)}) does not match the expected number ({expected_num_features}).")

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
