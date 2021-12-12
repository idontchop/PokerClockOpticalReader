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

![bounding boxes](https://github.com/idontchop/PokerClockOpticalReader/blob/main/renders/bounding.png)

