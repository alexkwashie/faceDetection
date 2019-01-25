import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#search haarcascades to use more detections

img = cv2.imread("people.jpg")

#creat grayscale image
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#resize img
re = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))

#add a face search area size
faces = face_cascade.detectMultiScale(gray_img,
scaleFactor = 1.05,
minNeighbors = 12)

#draw face detected area
for x, y, w, h in faces:
  img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,225,0),2)

print(type(faces)) #<class 'numpy.ndarray'>
print(faces)

cv2.imshow("Gray", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


