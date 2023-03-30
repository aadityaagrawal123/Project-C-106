import cv2

body_detection = cv2.CascadeClassifier("haarcascade_fullbody.xml")

vid = cv2.VideoCapture('walking.avi')

while (True):
    rect, frame = vid.read()
    grey_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    people = body_detection.detectMultiScale(grey_img, 1.1, 6)
    
    for (x,y,w,h) in people:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow("Output Window", frame)
    if cv2.waitKey(1) == 32:
        break

vid.release()
cv2.destroyAllWindows()
