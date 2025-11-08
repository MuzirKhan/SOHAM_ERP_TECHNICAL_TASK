import argparse
from analyzer import CCTVAnalyzer
from utils import setup_logging

# --- Configuration ---
LOG_FILE = "activity_log.csv"
SCREENSHOT_DIR = "screenshots"
ALERT_DURATION_SECONDS = 30

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Professional CCTV Activity Automation System.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file (e.g., sample.mp4).")
    args = parser.parse_args()

    setup_logging(LOG_FILE, SCREENSHOT_DIR)

    analyzer = CCTVAnalyzer(
        alert_duration_seconds=ALERT_DURATION_SECONDS,
        log_file=LOG_FILE,
        screenshot_dir=SCREENSHOT_DIR,
        target_fps=60
    )

    final_stats = analyzer.analyze_video(args.video)

    if final_stats:
        analyzer.print_final_report(final_stats)