import pyautogui
import time
for i in range(100):
    pyautogui.moveTo(1395, 536)
    pyautogui.click(button='right')
    pyautogui.moveTo(1475, 640)
    pyautogui.click(button='left')
    pyautogui.moveTo(1701, 568)
    pyautogui.click(button='left')
    time.sleep(1)
    pyautogui.scroll(-5)
    time.sleep(2)
