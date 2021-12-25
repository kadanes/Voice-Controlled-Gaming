import time
import keyboard
import time

def pressKeyMario(command):
        
    valid_commands = ["left", "right", "down"]
    if command in valid_commands:
        keyboard.press(command)
        time.sleep(1.5)
        keyboard.release(command)
 
    elif command == "go":
        keyboard.press("right+X")
        time.sleep(1.5)
        keyboard.release("right+X")
    elif command == "up":
        keyboard.press("X")
        time.sleep(1.5)
        keyboard.release("X")

def pressKeyTetris(command):

    valid_commands = ["left", "right", "down"]
    if command in valid_commands:
        keyboard.press(command)
        time.sleep(0.5)
        keyboard.release(command)

  
