import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch


def run_post_snap_analysis(play_id):
    base_dir = Path(__file__).resolve().parent
    tracking_model_path = os.path.join(base_dir, "tracking_model", "weights", "best.pt")
    role_model_path = os.path.join(base_dir, "role_model", "weights", "best.pt")
    video_path = os.path.join(base_dir, "testing", str(play_id), "post_snap.mp4")

    class_colors = {
        "QB": (0, 0, 255),
        "RB": (203, 192, 255),
        "WR": (0, 100, 0),
        "TE": (255, 0, 255),
        "OL": (235, 206, 135),
        "DEF": (0, 255, 255),
        "REF": (128, 128, 128),
        "UNK": (255, 255, 255)
    }

    FIELD_WIDTH = 500
    FIELD_HEIGHT = 700
    CIRCLE_RADIUS = 15
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    TRACKED_ROLES = {"WR", "RB", "TE"}

    #testing GPU
    role_model = YOLO(role_model_path).to('cuda')
    tracking_model = YOLO(tracking_model_path).to('cuda')
    

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def draw_paths(field, position_history, class_colors, role_filter):
        for t_id, history in position_history.items():
            if len(history) < 2:
                continue
            role = history[0][2]
            if role not in role_filter:
                continue
            points = np.array([(x, y) for x, y, _ in history], dtype=np.int32)
            color = class_colors.get(role, (150, 150, 150))
            cv2.polylines(field, [points], isClosed=False, color=color, thickness=2)

    if not os.path.exists(video_path):
        print("Video not found.")
        return

    cap = cv2.VideoCapture(video_path)

    tracking_id_to_role = {}
    lost_id_memory = {}
    valid_ids = set()
    position_history = {}
    frame_counter = 0

    locked_unique_xs = None
    locked_los_x_bev = None
    locked_avg_ol_y = None

    #better weights for deepsort
    embedder_weights_path = base_dir / "deep_sort_weights" / "osnet_ain_x1_0_wtsonly.pth"

    tracker = DeepSort(
        max_age=30,
        n_init=2,
        embedder='torchreid',
        embedder_model_name='osnet_ain_x1_0',
        embedder_wts= str(embedder_weights_path),
        half=True,
        bgr=True,

    )
   
    import torch

    if torch.cuda.is_available():
        tracker.embedder.model.model = tracker.embedder.model.model.cuda()
        print("DeepSORT embedder moved to:", next(tracker.embedder.model.model.parameters()).device)



    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            break

        role_results = role_model.predict(frame, conf=0.4, verbose=False)[0]
        role_boxes = role_results.boxes.xyxy.cpu().numpy()
        role_classes = [role_model.names[int(cls)] for cls in role_results.boxes.cls]

        det_results = tracking_model.predict(frame, conf=0.5, verbose=False)[0]
        det_boxes = det_results.boxes.xyxy.cpu().numpy()
        det_confs = det_results.boxes.conf.cpu().numpy()

        detections = [
            [(x1, y1, x2 - x1, y2 - y1), conf, "player"]
            for (x1, y1, x2, y2), conf in zip(det_boxes, det_confs)
        ]
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            t_id = track.track_id
            t_box = track.to_ltrb()
            best_iou = 0
            best_role = "UNK"
            for r_box, r_class in zip(role_boxes, role_classes):
                score = iou(t_box, r_box)
                if score > best_iou:
                    best_iou = score
                    best_role = r_class
            if best_iou > 0.3:
                tracking_id_to_role[t_id] = best_role
                valid_ids.add(t_id)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        results = tracking_model.predict(frame, conf=0.5, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        detections = [
            [(x1, y1, x2 - x1, y2 - y1), conf, "player"]
            for (x1, y1, x2, y2), conf in zip(boxes, confs)
        ]
        tracks = tracker.update_tracks(detections, frame=frame)

        field = 255 * np.ones((FIELD_HEIGHT, FIELD_WIDTH, 3), dtype=np.uint8)
        img_h, img_w = frame.shape[:2]

        if locked_unique_xs is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 130, 183)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=80, maxLineGap=10)
            x_positions = []
            if lines is not None:
                for line in lines:
                    x1_, y1_, x2_, y2_ = line[0]
                    dx = x2_ - x1_
                    dy = y2_ - y1_
                    if abs(dy / dx) if dx != 0 else float('inf') > 0.6 and np.hypot(dx, dy) > 320:
                        x_positions.append((x1_ + x2_) / 2)
            unique_xs = []
            for x in sorted(x_positions):
                if not unique_xs or abs(x - unique_xs[-1]) > 25:
                    unique_xs.append(x)
            if len(unique_xs) < 5:
                continue
            locked_unique_xs = unique_xs

            ol_x_bevs = []
            ol_y_bevs = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                t_id = track.track_id
                if t_id not in valid_ids:
                    continue
                role = tracking_id_to_role.get(t_id, "UNK")
                if role != "OL":
                    continue
                x1_, y1_, x2_, y2_ = map(int, track.to_ltrb())
                cx_ = (x1_ + x2_) / 2
                cy_ = (y1_ + y2_) / 2
                x_bev = int((cx_ / img_w) * FIELD_WIDTH)
                y_bev = int((cy_ / img_h) * FIELD_HEIGHT)
                ol_x_bevs.append(x_bev)
                ol_y_bevs.append(y_bev)
            if not ol_x_bevs:
                continue
            locked_los_x_bev = int(np.mean(ol_x_bevs))
            locked_avg_ol_y = int(np.mean(ol_y_bevs))

        for track in tracks:
            if not track.is_confirmed():
                continue
            t_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            box = [x1, y1, x2, y2]

            if t_id not in tracking_id_to_role:
                for lost_id, (lost_box, lost_role, last_seen) in lost_id_memory.items():
                    if frame_counter - last_seen > 30:
                        continue
                    if iou(box, lost_box) > 0.5:
                        tracking_id_to_role[t_id] = lost_role
                        break

            role = tracking_id_to_role.get(t_id, "UNK")
            lost_id_memory[t_id] = (box, role, frame_counter)

            if t_id not in valid_ids:
                continue

            x_bev = int((cx / img_w) * FIELD_WIDTH)
            y_bev = int((cy / img_h) * FIELD_HEIGHT)

            if t_id not in position_history:
                position_history[t_id] = []
            position_history[t_id].append((x_bev, y_bev, role))

            color = class_colors.get(role, (255, 255, 255))
            cv2.circle(field, (x_bev, y_bev), CIRCLE_RADIUS, color, -1)
            cv2.putText(field, f"{role} #{t_id}", (x_bev - 15, y_bev + 5), FONT, 0.5, (0, 0, 0), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{role} #{t_id}", (x1, y1 - 10), FONT, 0.6, color, 2)

        draw_paths(field, position_history, class_colors, TRACKED_ROLES)

        cv2.line(field, (locked_los_x_bev, 0), (locked_los_x_bev, FIELD_HEIGHT), (255, 0, 0), 2)
        cv2.putText(field, "LOS", (locked_los_x_bev + 5, 15), FONT, 0.5, (255, 0, 0), 1)
        cv2.line(field, (0, locked_avg_ol_y), (FIELD_WIDTH, locked_avg_ol_y), (150, 150, 150), 2)
        cv2.putText(field, "OL Y", (5, locked_avg_ol_y - 5), FONT, 0.5, (100, 100, 100), 1)

        lost_id_memory = {
            t_id: (box, role, last_seen)
            for t_id, (box, role, last_seen) in lost_id_memory.items()
            if frame_counter - last_seen <= 30
        }

        try:
            resized_frame = cv2.resize(frame, (FIELD_WIDTH * 2, FIELD_HEIGHT))
            bev_resized = cv2.resize(field, (FIELD_WIDTH, FIELD_HEIGHT))
            combined = cv2.hconcat([resized_frame, bev_resized])
        except Exception:
            continue

        yield combined

    cap.release()
   

if __name__ == "__main__":
    for frame in run_post_snap_analysis(play_id=26):
        cv2.imshow("Post-Snap Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  


