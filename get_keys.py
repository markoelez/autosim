from pynput import keyboard

keys = []

def on_press(key):
	try:
		# print('{} pressed'.format(key.char))
		if key.char.upper() not in keys:
			# print(key.char.upper())
			keys.append(key.char.upper())
			print(keys)
	except AttributeError:
		pass

def on_release(key):
	if key == keyboard.Key.esc:
	# Stop listener if 'esc' is pressed
		return False
	try:
		if key.char.upper() in keys:
			keys.remove(key.char.upper())
	except AttributeError:
		pass

def listen_keys():
	# Collect events until released
	with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
		listener.join()
	print(keys)

def key_check():
	
	return keys