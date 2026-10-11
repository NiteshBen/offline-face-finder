"""
Offline Face Finder - Main Entry Point
Robust launcher with crash reporting for standalone execution.
"""
import sys
import os
import traceback

def show_fatal_error(error_msg: str):
    log_path = "crash_log.txt"
    try:
        app_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(app_dir, "crash_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(error_msg)
    except Exception:
        pass

    # Pop up native Windows message box so user gets immediate visual feedback
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            f"Offline Face Finder could not start:\n\n{error_msg[:300]}...\n\nFull details saved to:\n{log_path}\n\n(Tip: Ensure Microsoft Visual C++ 2015-2022 Redistributable is installed).",
            "Offline Face Finder - Startup Error",
            0x10  # MB_ICONERROR
        )
    except Exception:
        print(f"Startup Error:\n{error_msg}", file=sys.stderr)

if __name__ == "__main__":
    try:
        from app_gui import OfflineFaceFinderApp
        app = OfflineFaceFinderApp()
        app.mainloop()
    except Exception:
        tb = traceback.format_exc()
        show_fatal_error(tb)
        sys.exit(1)
