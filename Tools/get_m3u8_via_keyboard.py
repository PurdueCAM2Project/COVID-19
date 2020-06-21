import webbrowser
import keyboard
import time
import pyperclip
import re
​
def getLink():
	URL = 'https://www.skylinewebcams.com/en/webcam/italia/veneto/venezia/piazza-san-marco.html'
	#webbrowser.open(URL, new=0, autoraise=True)
	webbrowser.get('firefox').open(URL)
	time.sleep(5)
	keyboard.press_and_release('command+u')
	time.sleep(3)
	keyboard.press_and_release('command+a')
	time.sleep(3)
	keyboard.press_and_release('command+c')
	dochtml = pyperclip.paste()
	link = re.search('source:"(.+)parentId', dochtml).group(1)
	print(link)
	link = link[0:len(link)-2]
	print(link)
	return link
​
getLink()
