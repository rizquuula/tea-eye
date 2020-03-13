import time
import pyautogui


# Save mouse position
(x, y) = pyautogui.position()
print(x,y)
# Your automated click
# time.sleep(1000)
pyautogui.click(200, 300)
(x, y) = pyautogui.position()
print(x,y)
pyautogui.moveTo(200,350)
(x, y) = pyautogui.position()
print(x,y)
# time.sleep(1)
# Move back to where the mouse was before click
pyautogui.moveTo(x, y)
print(x,y)
pyautogui.moveTo(100, 200, 2)
# pyautogui.displayMousePosition()