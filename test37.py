import cv2
import numpy as np


def detect_gender(box_tracks, frame1, boy, girl):
    for i in box_tracks:
        (x, y, w, h) = i
        # gender
        headImg = frame1[y: y + 20, x: x + w]
        n = 0
        theight, twidth = headImg.shape[:2]
        for i in range(theight):
            for j in range(twidth):
                [r, g, b] = headImg[i, j]
                if r < 80 and g < 80 and b < 80:
                    n += 1
        print(f'------------n:{n}')
        cv2.rectangle(frame1, (x, y), (x + w, y + 30), (255, 255, 0), 2)
        if(female(n, frame1, x, y) == True):
            girl += 1
        if(female(n, frame1, x, y) == False):
            boy += 1
        
        return boy, girl
def female(n, frame1, x , y):
    if n >= 150:
        cv2.putText(frame1, 'female', (x - 5, y + 15), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 0), 2)
        return True
    else:
        cv2.putText(frame1, 'male', (x - 5, y + 15), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 0), 2)
        return False
            
