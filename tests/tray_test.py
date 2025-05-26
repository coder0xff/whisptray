import pystray
from PIL import Image, ImageDraw
import os

# --- Global Variables (minimal for testing) ---
tray_icon_active = True # To control the icon's loop, analogous to dictation_active

# --- Helper Functions ---
def create_test_image(width, height, color1, color2):
    """Creates a simple image for the tray icon."""
    image = Image.new("RGB", (width, height), color1)
    dc = ImageDraw.Draw(image)
    dc.rectangle((width // 2, 0, width, height // 2), fill=color2)
    dc.rectangle((0, height // 2, width // 2, height), fill=color2)
    return image

# --- Tray Icon Callbacks ---
def on_test_action(icon, item):
    global tray_icon_active
    print(f"[DEBUG TRAY] Test action clicked. Current item: {item}")
    # Example action: toggle a state or print something
    tray_icon_active = not tray_icon_active
    print(f"[DEBUG TRAY] tray_icon_active state: {tray_icon_active}")
    icon.update_menu() # Update menu if checked state depends on this

def on_exit_tray(icon, item):
    print("[DEBUG TRAY] Exit action clicked.")
    icon.stop()
    print("[DEBUG TRAY] Icon stopped.")

# --- Main Setup ---
def main_tray_test():
    global app_icon # Keep a reference if needed, though pystray handles its own loop
    print("[DEBUG TRAY] main_tray_test started.")

    icon_image = create_test_image(64, 64, "red", "white") # Different color for test
    
    menu = pystray.Menu(
        pystray.MenuItem(
            "Test Action",
            on_test_action,
            # Example of a checkbox based on a global state
            checked=lambda item: tray_icon_active 
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(
            "Exit Test Tray",
            on_exit_tray
        )
    )
    
    app_icon = pystray.Icon("test_tray_app", icon_image, "Test Tray", menu)
    
    print("[DEBUG TRAY] pystray.Icon created. Calling app_icon.run().")
    try:
        app_icon.run()
    except Exception as e:
        print(f"[ERROR TRAY] Exception in app_icon.run(): {e}")
    finally:
        print("[DEBUG TRAY] app_icon.run() finished or exited.")

if __name__ == "__main__":
    print("[DEBUG TRAY] Starting tray_test.py...")
    # It's good practice to ensure DISPLAY is set for GUI apps on Linux
    if os.name == 'posix' and not os.environ.get("DISPLAY"): # More generic check for posix
        print("Error: DISPLAY environment variable not set. GUI cannot be displayed.")
        print("Please ensure you are running this in a graphical environment.")
    else:
        main_tray_test() 