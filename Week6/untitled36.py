
"""# Import libraries and define functions"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_and_convert_to_binary(image_path,thresh_hold):
    image = cv2.imread(image_path)
    image_copy = image.copy()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(image,thresh_hold,255,cv2.THRESH_BINARY)
    return image_copy,binary

def imshow_mask(dilation,mask_label,labels):
    mask = np.zeros_like(dilation, dtype=np.uint8)
    for label in labels:
        mask[mask_label==label] = 255
    plt.imshow(mask, cmap='gray')
    plt.show()
    return mask

def plot_result(image,mask,name):
    mask, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, 0, (0,255,0), 8)
    plt.imshow(image[...,::-1])
    cv2.imwrite(name+".jpg",image)

"""# Image 1"""

#image 1
image_path1 = 'homework-data/01.jpg'

image1,binary = read_and_convert_to_binary(image_path1,150)

image1.shape

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
dilation = cv2.dilate(opening,kernel,iterations=100)
dilation = cv2.erode(dilation,kernel,iterations=100)
plt.imshow(dilation,cmap='gray')

nlabel,labels,stats,centroids = cv2.connectedComponentsWithStats(dilation,connectivity=8)
for i in range(1, nlabel):
    print(stats[i, cv2.CC_STAT_HEIGHT])

mask = imshow_mask(dilation,labels,[4])
plot_result(image1,mask,'result_01')

"""# Image2"""

#image2
image2_path = 'homework-data/02.jpg'

image2,binary2 = read_and_convert_to_binary(image2_path,110)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(binary2, cv2.MORPH_OPEN, kernel)
dilation = cv2.erode(opening, kernel, iterations=10)
dilation = 255 - dilation

nlabel,labels,stats,centroids = cv2.connectedComponentsWithStats(dilation,connectivity=8)
for i in range(1, nlabel):
    print(stats[i, cv2.CC_STAT_HEIGHT])

mask = imshow_mask(dilation,labels,[93,94,95,96,97])
mask = cv2.dilate(mask, kernel, iterations=50)
plt.imshow(mask, cmap='gray')

plot_result(image2,mask,'result_02')

"""# Image3"""

image3_path = 'homework-data/03.jpg'

image3,binary3 = read_and_convert_to_binary(image3_path,120)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(binary3, cv2.MORPH_OPEN, kernel)
dilation = cv2.erode(opening, kernel, iterations=15)
dilation = cv2.dilate(dilation, kernel, iterations=15)
dilation = 255 - dilation

nlabel,labels,stats,centroids = cv2.connectedComponentsWithStats(dilation,connectivity=8)
for i in range(1, nlabel):
    print(stats[i, cv2.CC_STAT_HEIGHT])

mask = imshow_mask(dilation,labels,[28])

plot_result(image3,mask,'result_03')

"""# Image4"""

image4_path = 'homework-data/04.jpg'

image4,binary4 = read_and_convert_to_binary(image4_path,110)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(binary4, cv2.MORPH_OPEN, kernel)
dilation = cv2.erode(opening, kernel, iterations=15)
dilation = cv2.dilate(dilation, kernel, iterations=4)
dilation = 255 - dilation
dilation = cv2.erode(dilation, kernel, iterations=1)

plt.imshow(dilation)

nlabel,labels,stats,centroids = cv2.connectedComponentsWithStats(dilation,connectivity=8)
for i in range(1, nlabel):
    print(stats[i, cv2.CC_STAT_HEIGHT])

mask = imshow_mask(dilation,labels,[37])

mask = cv2.dilate(mask, kernel, iterations=100)
mask = cv2.erode(mask, kernel, iterations=100)

plot_result(image4,mask,'result_04')

"""# Image 5"""

image5_path = 'homework-data/05.jpg'

image5,binary5 = read_and_convert_to_binary(image5_path,120)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(binary5, cv2.MORPH_OPEN, kernel)
dilation = cv2.erode(opening, kernel, iterations=15)
dilation = cv2.dilate(dilation, kernel, iterations=4)

nlabel,labels,stats,centroids = cv2.connectedComponentsWithStats(dilation,connectivity=8)
for i in range(1, nlabel):
    print(stats[i, cv2.CC_STAT_HEIGHT])

mask = imshow_mask(dilation,labels,[127])

mask = cv2.dilate(mask, kernel, iterations=100)
mask = cv2.erode(mask, kernel, iterations=100)

plot_result(image5,mask,'result_05')

