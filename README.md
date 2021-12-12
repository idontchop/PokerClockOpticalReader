# Poker Clock Optical Reader

Python work attempting to read a poker tournament clock, eventually allowing an app to read the clock via a camera.

## Libraries used

- OpenCv
- Tesseract

## Initial Strategy

Basic work done so far to threshold images and run them through Tesseract basic settings and user patterns to detect the '.....' lines with places paid. Getting the price pool and total chips seems possible but the periods as well as the closeness to the border causes Tesseract to misread the place values.

Further work will involve stripping all lines out of the image and / or training a more specific Tensorflow model.

```python
from pytesseract import Output
custom_config = r'--oem 3 --psm 6 --user-patterns patterns.txt'
    #-c tessedit_char_whitelist=PrizePool$123456789stndrdth.,:'
details = pytesseract.image_to_data(f, output_type=Output.DICT, config=custom_config, lang="eng")

print(len(details['text']))
for i in range(len(details['level'])):
    if (int(details['conf'][i]) >= 0):
        print(str(i) + ": " + str(details['conf'][i]) + "% " + str(details['text'][i]))
```

![bounding boxes](https://github.com/idontchop/PokerClockOpticalReader/blob/main/renders/pretty.png)

Work to find all rectangles in the tournament board to extract the places section was not successful with OpenCv

```python
import numpy as np

newImg = np.copy(image)

contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#for c in contours:
#    approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
#    if (len(approx) == 4):
#        cv2.drawContours(newImg, [approx], 0, (0,0,0),5)

for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    if w*h > 100000:
        print(x,y,w,h)
        newImg = cv2.rectangle(newImg, (x,y), (x+w, y+h), (0,255,0),4)

```

![Rectangles Bounding Boxes](https://github.com/idontchop/PokerClockOpticalReader/blob/main/renders/bounding.png)
