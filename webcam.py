# import cv2
# import numpy as np
# import tensorflow as tf
# import json

# # Load trained model
# model = tf.keras.models.load_model(
#     "models/SignSpeak_finetuned.keras",
#     custom_objects={
#         "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
#     }
# )

# # Load class names
# with open("class_names.json", "r") as f:
#     class_names = json.load(f)

# # Open webcam
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     # Resize frame to model input size
#     image = cv2.resize(frame, (224, 224))

#     # OpenCV uses BGR, convert to RGB
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Add batch dimension
#     image = np.expand_dims(image, axis=0)

#     # Predict
#     prediction = model.predict(image, verbose=0)[0]

#     predicted_index = np.argmax(prediction)
#     predicted_letter = class_names[predicted_index]
#     confidence = prediction[predicted_index]

#     # Display prediction
#     text = f"{predicted_letter} ({confidence * 100:.1f}%)"

#     cv2.putText(
#         frame,
#         text,
#         (30, 60),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1.5,
#         (0, 255, 0),
#         3
#     )

#     cv2.imshow("SignSpeak", frame)

#     # Press Q to quit
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()





import cv2
import numpy as np
import tensorflow as tf
import json
import mediapipe as mp
from collections import deque

# =========================
# Load Model
# =========================

model = tf.keras.models.load_model(
    "models/SignSpeak_finetuned.keras",
    custom_objects={
        "preprocess_input":
        tf.keras.applications.mobilenet_v2.preprocess_input
    }
)

with open("class_names.json", "r") as f:
    class_names = json.load(f)


# =========================
# MediaPipe Hands
# =========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================
# Prediction Smoothing
# =========================

prediction_history = deque(maxlen=7)

CONFIDENCE_THRESHOLD = 0.60


# =========================
# Letter History
# =========================

letter_history = deque(maxlen=8)


# =========================
# Webcam
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()


while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    prediction_text = "No hand detected"

    if results.multi_hand_landmarks:

        xs = []
        ys = []

        # =========================
        # Collect Hand Landmarks
        # =========================

        for hand_landmarks in results.multi_hand_landmarks:

            for landmark in hand_landmarks.landmark:

                xs.append(
                    int(landmark.x * w)
                )

                ys.append(
                    int(landmark.y * h)
                )

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        # =========================
        # Bounding Box
        # =========================

        x1 = max(0, min(xs))
        y1 = max(0, min(ys))

        x2 = min(w, max(xs))
        y2 = min(h, max(ys))

        padding = 20

        x1 = max(
            0,
            x1 - padding
        )

        y1 = max(
            0,
            y1 - padding
        )

        x2 = min(
            w,
            x2 + padding
        )

        y2 = min(
            h,
            y2 + padding
        )

        crop = frame[
            y1:y2,
            x1:x2
        ]

        if crop.size != 0:

            # =========================
            # Prepare Image
            # =========================

            image = cv2.resize(
                crop,
                (224, 224)
            )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            image = np.expand_dims(
                image,
                axis=0
            )

            # =========================
            # Model Prediction
            # =========================

            prediction = model.predict(
                image,
                verbose=0
            )[0]

            prediction_history.append(
                prediction
            )

            # Average recent predictions
            average_prediction = np.mean(
                prediction_history,
                axis=0
            )

            predicted_index = np.argmax(
                average_prediction
            )

            predicted_letter = class_names[
                predicted_index
            ]

            confidence = average_prediction[
                predicted_index
            ]

            # =========================
            # Confidence Check
            # =========================

            if confidence >= CONFIDENCE_THRESHOLD:

                prediction_text = (
                    f"{predicted_letter} "
                    f"{confidence * 100:.1f}%"
                )

                # Add to history only if
                # prediction is confident
                if (
                    len(letter_history) == 0
                    or letter_history[-1]
                    != predicted_letter
                ):
                    letter_history.append(
                        predicted_letter
                    )

            else:

                prediction_text = (
                    "Uncertain"
                )

            # =========================
            # Bounding Box
            # =========================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    else:

        prediction_history.clear()

    # =========================
    # UI Background
    # =========================

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (w, 120),
        (0, 0, 0),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0
    )

    # =========================
    # Title
    # =========================

    cv2.putText(
        frame,
        "SignSpeak",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =========================
    # Prediction
    # =========================

    cv2.putText(
        frame,
        prediction_text,
        (25, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (0, 255, 0),
        3
    )

    # =========================
    # Letter History
    # =========================

    history_text = "History: "

    if letter_history:

        history_text += " ".join(
            letter_history
        )

    else:

        history_text += "-"

    cv2.putText(
        frame,
        history_text,
        (25, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =========================
    # Quit Instruction
    # =========================

    cv2.putText(
        frame,
        "Press Q to quit",
        (w - 190, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1
    )

    # =========================
    # Display
    # =========================

    cv2.imshow(
        "SignSpeak",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# Cleanup
# =========================

cap.release()
hands.close()
cv2.destroyAllWindows()
