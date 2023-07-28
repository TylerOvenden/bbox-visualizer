import json
import imutils

import bbox_visualizer as bbv
import cv2
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

img = cv2.imread(r"C:\Users\vleen\Documents\GitHub\bbox-visualizer\images\pexels.jpg")
annotation = json.load(open(r"C:\Users\vleen\Documents\GitHub\bbox-visualizer\demo\source_single.json"))
img = imutils.resize(img,
                       width=min(600, img.shape[1]))

(regions, _) = hog.detectMultiScale(img, 
                                    winStride=(2, 2),
                                    padding=(4, 4),
                                    scale=1.05)
   
# Drawing the regions in the Image
for (x, y, w, h) in regions:
    cv2.rectangle(img, (x, y), 
                  (x + w, y + h), 
                  (0, 0, 255), 2)
  
# Showing the output Image
cv2.imshow("Image", img)
cv2.waitKey(0)
'''
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

points = annotation['shapes'][0]['points']
label = annotation['shapes'][0]['label']
(xmin, ymin), (xmax, ymax) = points
bbox = [xmin, ymin, xmax, ymax]


img = bbv.draw_rectangle(img, bbox)
img = bbv.add_label(img, label, bbox, top=True)
if img is not None:
    resized_img = cv2.resize(img, (800, 533))
    print("Image shape:", resized_img.shape) 
    cv2.imshow("Resized Image", resized_img)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
'''