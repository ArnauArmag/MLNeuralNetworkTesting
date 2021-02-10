import cv2 as cv

'''
img=cv.imread('snake.jpeg')

cv.imshow('Snake',img)
'''
capture=cv.VideoCapture(0)
counter=0

while True:
    counter=counter+1
    isTrue, frame=capture.read()
    print(counter)
    #cv.imshow('Video',frame)
    print(frame.shape)
    #if cv.waitKey(20) & 0xFF==ord('d'):
    if cv.waitKey(20) & counter>2:
        break

capture.release()
cv.destroyAllWindows()
