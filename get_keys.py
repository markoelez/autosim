from pynput import keyboard

keys = []

def on_press(key):
	try:
		if key.char.upper() not in keys:
			keys.append(key.char.upper())
	except AttributeError:
		pass

def on_release(key):
	# if key == keyboard.Key.esc:
	# 	return False
	try:
		if key.char.upper() in keys:
			keys.remove(key.char.upper())
	except AttributeError:
		pass

def listen_keys():
	# Collect events until released
	with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
		listener.join()

def key_check():
	return keys