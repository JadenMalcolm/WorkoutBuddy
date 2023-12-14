import pyautogui
import time

while True:
    x, y = pyautogui.position()
    time.sleep(1)
    print(f"X: {x}, Y: {y}")
