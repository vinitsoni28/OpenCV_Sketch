import cv2
import numpy as np

img = "img/mountain.jpg"

img_obj = cv2.imread(img)
# print(img_obj.shape)

scale_per = 0.20
width = int(img_obj.shape[1] * scale_per)
height = int(img_obj.shape[0] * scale_per)

dimension = (width, height)
resized = cv2.resize(img_obj, dimension, interpolation = cv2.INTER_AREA)

# cv2.imshow("Original Image", img_obj)
cv2.imshow("Resized Image", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened = cv2.filter2D(resized, -1, sharpening)

# cv2.imshow("Sharp", sharpened)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
object_detection = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)

# cv2.imshow("Gray", gray)
# cv2.imshow("Object Detection", object_detection)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

inv = 255-gray
gauss = cv2.GaussianBlur(inv, ksize = (15, 15), sigmaX=0, sigmaY=0)

sketch = cv2.divide(gray, 255-gauss, scale=256)
cv2.imshow("Sketch", sketch)
cv2.waitKey(0)
cv2.destroyAllWindows()




