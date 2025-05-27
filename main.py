import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from play_prediction.predict_play import predict_single_play
from pre_snap.pre_snap_analysis import run_pre_snap_analysis
from post_snap.snap_analysis import run_post_snap_analysis

#Play Ids for visualization, 26-50 are valid. 
start_id = 26
end_id = 50

base_dir = Path(__file__).resolve().parent

def render_prediction_header(play_id, prediction_text, width):
    header = 255 * np.ones((60, width, 3), dtype=np.uint8)
    display_text = f"Play ID: {play_id} | {prediction_text}"
    cv2.putText(header, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return header


def render_footer(play_id, width):
    df = pd.read_csv(base_dir / "test_game_data.csv")
    row = df[df['id'] == play_id]
    outcome = row['play_type'].values[0] if not row.empty else "Unknown"
    footer = 255 * np.ones((60, width, 3), dtype=np.uint8)
    display_text = f"Actual Outcome: {outcome}"
    cv2.putText(footer, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return footer


def play_type_prediction(play_id):
    df = pd.read_csv(base_dir / "test_game_data.csv")
    feature_columns = [
        "down", "ydstogo", "half_seconds_remaining", "score_differential",
        "yardline_100", "pass_pct_last_20", "pass_pct_diff_10_vs_40"
    ]
    play = df[df['id'] == play_id].dropna(subset=feature_columns)
    return predict_single_play(play.iloc[0]) if not play.empty else "Missing data"


def run_dashboard_for_play(play_id):
    width = 960
    header_height = 60
    pre_snap_height = 360
    post_snap_height = 360
    footer_height = 60
    border_thickness = 3
    total_height = (header_height + border_thickness + pre_snap_height + border_thickness +
                    post_snap_height + border_thickness + footer_height)

    prediction_text = play_type_prediction(play_id)
    if not prediction_text:
        return

    pre_snap_stream = run_pre_snap_analysis(play_id)
    post_snap_stream = None
    last_pre_frame = None
    footer = render_footer(play_id, width)

    for _ in range(10000):
        dashboard = np.ones((total_height, width, 3), dtype=np.uint8) * 255
        dashboard[0:header_height, :] = render_prediction_header(play_id, prediction_text, width)
        dashboard[header_height:header_height + border_thickness, :] = 0

        if pre_snap_stream:
            try:
                pre_frame = next(pre_snap_stream)
                last_pre_frame = pre_frame
                pre_snap_resized = cv2.resize(pre_frame, (width, pre_snap_height))
                dashboard[header_height + border_thickness:header_height + border_thickness + pre_snap_height, :] = pre_snap_resized
            except StopIteration:
                pre_snap_stream = None
                post_snap_stream = run_post_snap_analysis(play_id)
        elif last_pre_frame is not None:
            pre_snap_resized = cv2.resize(last_pre_frame, (width, pre_snap_height))
            dashboard[header_height + border_thickness:header_height + border_thickness + pre_snap_height, :] = pre_snap_resized

        pre_bottom = header_height + border_thickness + pre_snap_height
        dashboard[pre_bottom:pre_bottom + border_thickness, :] = 0

        if post_snap_stream:
            try:
                post_frame = next(post_snap_stream)
                post_snap_resized = cv2.resize(post_frame, (width, post_snap_height))
                dashboard[pre_bottom + border_thickness:pre_bottom + border_thickness + post_snap_height, :] = post_snap_resized
            except StopIteration:
                post_snap_stream = None
        else:
            post_placeholder = np.ones((post_snap_height, width, 3), dtype=np.uint8) * 230
            cv2.putText(post_placeholder, "Post Snap Analysis", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
            dashboard[pre_bottom + border_thickness:pre_bottom + border_thickness + post_snap_height, :] = post_placeholder

        post_bottom = pre_bottom + border_thickness + post_snap_height
        dashboard[post_bottom:post_bottom + border_thickness, :] = 0
        dashboard[post_bottom + border_thickness:post_bottom + border_thickness + footer_height, :] = footer

        cv2.imshow("Dashboard", dashboard)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not pre_snap_stream and not post_snap_stream:
            break

    cv2.destroyAllWindows()

def main(): 
    for play_id in range(start_id, end_id + 1):
        run_dashboard_for_play(play_id)

if __name__ == "__main__":
    main()









