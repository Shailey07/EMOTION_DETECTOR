"""
Face Emotion Detection - Camera se Real-time
=============================================
Emotions detect karta hai: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral

Requirements install karo:
    pip install deepface opencv-python tf-keras

Run karo:
    python trained_model.py
"""

import cv2
from deepface import DeepFace

# Camera start karo
cap = cv2.VideoCapture(0)

# Emotion ke liye colors (BGR format)
EMOTION_COLORS = {
    "happy":    (0, 220, 100),
    "sad":      (200, 80,  20),
    "angry":    (0,   0, 220),
    "fear":     (180, 0, 180),
    "surprise": (0, 200, 220),
    "disgust":  (0, 150,  50),
    "neutral":  (160, 160, 160),
}

# Emotion ke liye emojis (display ke liye)
EMOTION_EMOJI = {
    "happy":    "HAPPY    :)",
    "sad":      "SAD      :(",
    "angry":    "ANGRY    >:(",
    "fear":     "FEAR     D:",
    "surprise": "SURPRISE :O",
    "disgust":  "DISGUST  :/",
    "neutral":  "NEUTRAL  :|",
}

print("=" * 45)
print("  FACE EMOTION DETECTION - CAMERA ACTIVE")
print("  Press 'Q' to quit")
print("=" * 45)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera nahi mili! Check karo.")
        break

    # Frame flip karo (mirror effect)
    frame = cv2.flip(frame, 1)
    display = frame.copy()

    try:
        # DeepFace se emotion analyze karo
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True
        )

        # Multiple faces support
        if not isinstance(results, list):
            results = [results]

        for result in results:
            # Dominant emotion
            dominant = result["dominant_emotion"].lower()
            emotions = result["emotion"]

            # Face region
            region = result.get("region", {})
            x = region.get("x", 10)
            y = region.get("y", 10)
            w = region.get("w", 200)
            h = region.get("h", 200)

            color = EMOTION_COLORS.get(dominant, (255, 255, 255))

            # Face ke around rectangle
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

            # Emotion label box (face ke upar)
            label = EMOTION_EMOJI.get(dominant, dominant.upper())
            label_bg_y1 = max(0, y - 35)
            label_bg_y2 = max(35, y)
            cv2.rectangle(display, (x, label_bg_y1), (x + w, label_bg_y2), color, -1)
            cv2.putText(
                display, label,
                (x + 6, label_bg_y2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
            )

            # Side panel - saari emotions ki percentage
            panel_x = 10
            panel_y_start = 30
            cv2.rectangle(display, (panel_x - 5, panel_y_start - 25),
                          (220, panel_y_start + len(emotions) * 28 + 5),
                          (20, 20, 20), -1)
            cv2.putText(display, "EMOTIONS %", (panel_x, panel_y_start - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Emotions ko score ke hisaab se sort karo
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

            for i, (emo, score) in enumerate(sorted_emotions):
                emo_color = EMOTION_COLORS.get(emo, (160, 160, 160))
                y_pos = panel_y_start + i * 28 + 18

                # Progress bar background
                cv2.rectangle(display,
                              (panel_x, y_pos - 13),
                              (210, y_pos + 3),
                              (50, 50, 50), -1)

                # Progress bar fill
                bar_width = int(score * 1.8)  # max ~180px for 100%
                if bar_width > 0:
                    cv2.rectangle(display,
                                  (panel_x, y_pos - 13),
                                  (panel_x + bar_width, y_pos + 3),
                                  emo_color, -1)

                # Emotion name aur score
                emo_text = f"{emo.upper()[:7]:<7} {score:5.1f}%"
                text_color = (255, 255, 255) if emo == dominant else (180, 180, 180)
                cv2.putText(display, emo_text,
                            (panel_x + 2, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            text_color, 1)

    except Exception as e:
        # Agar face nahi mila
        cv2.putText(display, "no face detected. Adjust lighting and make sure face is clear.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (100, 100, 255), 2)

    # Bottom me instruction
    h_frame = display.shape[0]
    cv2.rectangle(display, (0, h_frame - 30), (display.shape[1], h_frame), (20, 20, 20), -1)
    cv2.putText(display, "Press Q to Quit | Face Emotion Detector",
                (10, h_frame - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow("Face Emotion Detector", display)

    # Q press karo quit ke liye
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Detector is going to sleep...")
        break

cap.release()
cv2.destroyAllWindows()
print("Done. Camera is off.")