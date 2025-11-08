import cv2
from ultralytics import YOLO
from utils import log_event, frames_to_time_str, blur_faces

# YOLOv8 Class IDs
PERSON_CLASS_ID = 0
MOBILE_CLASS_ID = 67
LAPTOP_CLASS_ID = 63
KEYBOARD_CLASS_ID = 66


class CCTVAnalyzer:
    """Analyzes video for employee activity using YOLO and tracks cumulative time."""

    def __init__(self, alert_duration_seconds, log_file, screenshot_dir, target_fps=30):
        self.ALERT_DURATION_SECONDS = alert_duration_seconds
        self.LOG_FILE = log_file
        self.SCREENSHOT_DIR = screenshot_dir
        self.TARGET_FPS = target_fps
        self.FRAME_SKIP = 5
        self.OUTPUT_VIDEO_FILE = "output_video.mp4"

        self.off_camera_start_frame = -1
        self.mobile_in_hand_start_frame = -1  # <-- NEW: Start of mobile usage tracking
        self.frame_count = 0
        self.alert_duration_frames = 0
        self.last_person_present = False
        self.last_mobile_in_hand = False
        self.last_laptop_detected = False

        self.stats = {
            "total_video_frames": 0,
            "off_camera_frames": 0,
            "mobile_in_hand_frames": 0,
            "working_frames": 0,
            "current_fps": self.TARGET_FPS
        }

        self.model = YOLO('yolov8n.pt')

    def analyze_video(self, video_path):
        """Processes the video frame by frame and saves the output."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or self.TARGET_FPS
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.stats["current_fps"] = fps
        self.alert_duration_frames = int(fps * self.ALERT_DURATION_SECONDS)

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(self.OUTPUT_VIDEO_FILE, fourcc, fps, (frame_width, frame_height))
        if not video_writer.isOpened():
            print(f"Warning: VideoWriter failed to open. Output video will not be saved.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            self.stats["total_video_frames"] = self.frame_count + 1

            # --- A. Detection/Extraction (Heavy Operation - Only run every FRAME_SKIP frames) ---
            if self.frame_count % self.FRAME_SKIP == 0:
                results = self.model(frame, verbose=False)

                person_boxes = []
                mobile_in_hand_current = False
                laptop_keyboard_detected_current = False
                mobile_event_draw_boxes = False  # Flag just for drawing boxes on this frame

                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls == PERSON_CLASS_ID:
                            person_boxes.append(box.xyxy[0].cpu().numpy().astype(int))
                        elif cls in [LAPTOP_CLASS_ID, KEYBOARD_CLASS_ID]:
                            laptop_keyboard_detected_current = True

                # Check Mobile in Hand (Refined Proximity Check)
                for p_box in person_boxes:
                    px1, py1, px2, py2 = p_box
                    py_mobile_zone = py1 + int((py2 - py1) * 0.20)
                    p_center_x = (px1 + px2) / 2
                    p_width = px2 - px1

                    for r in results:
                        for box in r.boxes:
                            if int(box.cls[0]) == MOBILE_CLASS_ID:
                                mx1, my1, mx2, my2 = box.xyxy[0].cpu().numpy().astype(int)

                                m_center_x = (mx1 + mx2) / 2
                                m_center_y = (my1 + my2) / 2

                                horizontal_prox = abs(m_center_x - p_center_x) < (p_width * 0.6)
                                vertical_prox = m_center_y > py_mobile_zone
                                touches = (max(0, min(px2, mx2) - max(px1, mx1)) * max(0, min(py2, my2) - max(py1,
                                                                                                              my1))) > 0

                                if horizontal_prox and vertical_prox and touches:
                                    mobile_in_hand_current = True

                                    # Set draw flag (drawing happens after logic)
                                    if not mobile_event_draw_boxes:
                                        mobile_event_draw_boxes = True
                                    break
                        if mobile_event_draw_boxes: break
                    if mobile_event_draw_boxes: break

                    # Update the state variables
                self.last_person_present = len(person_boxes) > 0
                self.last_mobile_in_hand = mobile_in_hand_current
                self.last_laptop_detected = laptop_keyboard_detected_current

            # --- C. Tracking and Inference (Runs on EVERY frame) ---
            is_person_present = self.last_person_present
            mobile_in_hand_state = self.last_mobile_in_hand  # Current state of mobile usage
            laptop_keyboard_detected = self.last_laptop_detected
            current_activity = "N/A"

            # 1. LOGIC FOR MOBILE DURATION (New Debounce)
            if mobile_in_hand_state and self.mobile_in_hand_start_frame == -1:
                # Event STARTED: Log the start frame and take a screenshot
                self.mobile_in_hand_start_frame = self.frame_count
                # Call log event with 0 duration on START
                log_event(frame, "mobile_in_hand_start", 0, self.LOG_FILE, self.SCREENSHOT_DIR)
            elif not mobile_in_hand_state and self.mobile_in_hand_start_frame != -1:
                # Event ENDED: Log the final duration and reset
                duration_frames = self.frame_count - self.mobile_in_hand_start_frame
                duration_seconds = duration_frames / fps
                log_event(frame, "mobile_in_hand_end", duration_seconds, self.LOG_FILE, self.SCREENSHOT_DIR)
                self.mobile_in_hand_start_frame = -1

            # 2. HIERARCHICAL ACTIVITY INFERENCE (For Display and Cumulative Count)
            if is_person_present:
                self.off_camera_start_frame = -1
                status_color = (0, 255, 0)

                if mobile_in_hand_state:  # Use the current state
                    current_activity = "Using Mobile Phone"
                    self.stats["mobile_in_hand_frames"] += 1
                elif laptop_keyboard_detected:
                    current_activity = "Working on Laptop"
                    self.stats["working_frames"] += 1
                else:
                    current_activity = "Present (Idle/Other)"

            else:
                self.stats["off_camera_frames"] += 1
                current_activity = "Off-Camera/Missing"
                status_color = (0, 0, 255)

                # Missing Alert logic
                if self.off_camera_start_frame == -1:
                    self.off_camera_start_frame = self.frame_count

                duration_frames = self.frame_count - self.off_camera_start_frame

                if duration_frames >= self.alert_duration_frames:
                    duration_seconds = duration_frames / fps
                    log_event(frame, "person_missing_alert", duration_seconds, self.LOG_FILE, self.SCREENSHOT_DIR)
                    self.off_camera_start_frame = -1

                    # --- D. Display Status ---
            cv2.putText(frame, f"Activity: {current_activity}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color,
                        2)

            if is_person_present:
                alert_time_str = "00:00"
            else:
                duration_frames = self.frame_count - self.off_camera_start_frame
                duration_seconds = duration_frames / fps
                minutes = int(duration_seconds // 60)
                seconds = int(duration_seconds % 60)
                alert_time_str = f"{minutes:02}:{seconds:02}"

            alert_label = f"Missing Alert: {alert_time_str}/{self.ALERT_DURATION_SECONDS:02}s"
            cv2.putText(frame, alert_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # --- E. Loop Control & Saving ---
            display_frame = frame.copy()
            display_frame = blur_faces(display_frame)

            if video_writer.isOpened():
                video_writer.write(display_frame)

            cv2.imshow('CCTV Automation Monitor (Press "q" to exit)', frame)  # Show UNBLURRED Frame

            self.frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # --- F. Cleanup ---
        cap.release()
        cv2.destroyAllWindows()
        if video_writer.isOpened():
            video_writer.release()
            print(f"\nOutput video saved to: {self.OUTPUT_VIDEO_FILE}")

        return self.stats

    def print_final_report(self, stats):
        fps = stats["current_fps"]
        total_frames = stats["total_video_frames"]
        total_time_str = frames_to_time_str(total_frames, fps)

        print("\n" + "=" * 50)
        print("           CUMULATIVE ACTIVITY REPORT")
        print("=" * 50)
        print(f"Total Video Duration: {total_time_str}")
        print("-" * 50)

        off_camera_frames = stats["off_camera_frames"]
        off_camera_time_str = frames_to_time_str(off_camera_frames, fps)
        off_camera_percent = (off_camera_frames / total_frames * 100) if total_frames > 0 else 0
        print(f"Total Time Off Camera: {off_camera_time_str} ({off_camera_percent:.1f}%)")

        mobile_frames = stats["mobile_in_hand_frames"]
        mobile_time_str = frames_to_time_str(mobile_frames, fps)
        mobile_percent = (mobile_frames / total_frames * 100) if total_frames > 0 else 0
        print(f"Total Time Used Mobile: {mobile_time_str} ({mobile_percent:.1f}%)")

        working_frames = stats["working_frames"]
        working_time_str = frames_to_time_str(working_frames, fps)
        working_percent = (working_frames / total_frames * 100) if total_frames > 0 else 0
        print(f"Total Time Working:   {working_time_str} ({working_percent:.1f}%)")
        print("=" * 50)