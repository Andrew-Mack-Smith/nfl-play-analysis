import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

base_dir = Path(__file__).resolve().parent

def run_pre_snap_analysis(play_id):
    base_dir = Path(__file__).resolve().parent
    model_path = os.path.join(base_dir, 'model', 'weights', 'best.pt')
    video_path = os.path.join(base_dir, 'testing', str(play_id), 'pre_snap.mp4')

    model = YOLO(model_path)

    FIELD_WIDTH = 500
    FIELD_HEIGHT = 700
    CIRCLE_RADIUS = 15
    CONF_THRESH = 0.4
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    class_colors = {
        "QB": (0, 0, 255),
        "RB": (203, 192, 255),
        "WR": (0, 100, 0),
        "TE": (255, 0, 255),
        "OL": (235, 206, 135),
        "DEF": (0, 255, 255),
        "REF": (128, 128, 128)
    }

    if not os.path.isfile(video_path):
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    locked_unique_xs = None
    yardline_bev_xs = []
    locked_los_x_bev = None
    locked_avg_ol_y = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]
        results = model(frame, verbose = False)[0]

        player_points = []
        for box in results.boxes:
            if box.conf[0] < CONF_THRESH:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            player_points.append((label, x_center, y_center))

            color = class_colors.get(label, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10), FONT, 0.5, color, 2)

        if locked_unique_xs is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 130, 183)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=80, maxLineGap=10)
            x_positions = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    dx = x2 - x1
                    dy = y2 - y1
                    if abs(dy / dx) if dx != 0 else float('inf') > 0.6 and np.hypot(dx, dy) > 320:
                        x_positions.append((x1 + x2) / 2)

            unique_xs = []
            for x in sorted(x_positions):
                if not unique_xs or abs(x - unique_xs[-1]) > 25:
                    unique_xs.append(x)

            if len(unique_xs) < 5 or not player_points:
                continue

            yardline_bev_xs = [int((i + 1) * FIELD_WIDTH / (len(unique_xs) + 1)) for i in range(len(unique_xs))]
            locked_unique_xs = unique_xs

            ol_x_bevs = []
            ol_y_bevs = []
            for label, x, y in player_points:
                for i, (x_left, x_right) in enumerate(zip(unique_xs[:-1], unique_xs[1:])):
                    if x_left <= x <= x_right:
                        ratio = (x - x_left) / (x_right - x_left)
                        x_bev = int(yardline_bev_xs[i] + ratio * (yardline_bev_xs[i + 1] - yardline_bev_xs[i]))
                        break
                else:
                    x_bev = yardline_bev_xs[0] if x < unique_xs[0] else yardline_bev_xs[-1]

                y_bev = int((y / img_h) * FIELD_HEIGHT)

                if label == "OL":
                    ol_x_bevs.append(x_bev)
                    ol_y_bevs.append(y_bev)

            if not ol_x_bevs:
                continue

            locked_los_x_bev = int(np.mean(ol_x_bevs))
            locked_avg_ol_y = int(np.mean(ol_y_bevs))

            x_offset = (FIELD_WIDTH // 2) - locked_los_x_bev
            y_offset = (FIELD_HEIGHT // 2) - locked_avg_ol_y

            yardline_bev_xs = [x + x_offset for x in yardline_bev_xs]
            locked_los_x_bev += x_offset
            locked_avg_ol_y += y_offset

        field = 255 * np.ones((FIELD_HEIGHT, FIELD_WIDTH, 3), dtype=np.uint8)
        los_x_bev = locked_los_x_bev
        avg_ol_y = locked_avg_ol_y
        unique_xs = locked_unique_xs

        player_bevs = []
        for label, x, y in player_points:
            for i, (x_left, x_right) in enumerate(zip(unique_xs[:-1], unique_xs[1:])):
                if x_left <= x <= x_right:
                    ratio = (x - x_left) / (x_right - x_left)
                    x_bev = int(yardline_bev_xs[i] + ratio * (yardline_bev_xs[i + 1] - yardline_bev_xs[i]))
                    break
            else:
                x_bev = yardline_bev_xs[0] if x < unique_xs[0] else yardline_bev_xs[-1]

            y_bev = int((y / img_h) * FIELD_HEIGHT)
            player_bevs.append((label, x_bev, y_bev))

        offensive_xs = [x for label, x, y in player_bevs if label in {"OL", "QB"}]
        offense_on_left = np.mean(offensive_xs) < los_x_bev if offensive_xs else True

        cv2.line(field, (los_x_bev, 0), (los_x_bev, FIELD_HEIGHT), (255, 0, 0), 2)
        cv2.putText(field, "LOS", (los_x_bev + 5, 15), FONT, 0.5, (255, 0, 0), 1)
        cv2.line(field, (0, avg_ol_y), (FIELD_WIDTH, avg_ol_y), (150, 150, 150), 2)
        cv2.putText(field, "OL Y", (5, avg_ol_y - 5), FONT, 0.5, (100, 100, 100), 1)

        for label, x_bev, y_bev in player_bevs:
            if label in {"QB", "RB", "WR", "TE", "OL"}:
                x_bev = min(x_bev, los_x_bev - 5) if offense_on_left else max(x_bev, los_x_bev + 5)
            elif label == "DEF":
                x_bev = max(x_bev, los_x_bev + 5) if offense_on_left else min(x_bev, los_x_bev - 5)

            color = class_colors.get(label, (200, 200, 200))
            cv2.circle(field, (x_bev, y_bev), CIRCLE_RADIUS, color, -1)
            cv2.putText(field, label, (x_bev - 12, y_bev + 5), FONT, 0.4, (0, 0, 0), 1)

        resized_frame = cv2.resize(frame, (FIELD_WIDTH * 2, FIELD_HEIGHT))
        combined = cv2.hconcat([resized_frame, field])
        yield combined

    cap.release()
    

if __name__ == "__main__":
    for frame in run_pre_snap_analysis(play_id=26):
        cv2.imshow("Pre-Snap Viewer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    



