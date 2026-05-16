import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

img_origin = "messi.jpg"
img = cv2.imread(img_origin)
origin_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized_img= cv2.resize(gray_img,None, fx = 1.5, fy = 1.5, interpolation=cv2.INTER_LINEAR)

equalized = cv2.equalizeHist(gray_img)

def info_img(img,img_name= "Image"):
    if len(img.shape) == 3:
        height, width, channels = img.shape
    else:
        height, width = img.shape
        channels = 1
    print(f"Image size: {height}x{width} pixels ")
    print(f"Numbers of channels: {channels}")
    print(f"Datatype: {img.dtype}")
    print(f"Min pixel value: {img.min()}")
    print(f"Max pixels value: {img.max()}")

info_img(origin_img, "Original Image")
print("=======================")
info_img(gray_img, "Gray Image")
plt.figure(figsize=(13, 6))
# --- Original Image ---
plt.subplot(2,3,1)
plt.imshow(origin_img)
plt.title("Original Image")
plt.axis("off")

#---Gray Image-----
plt.subplot(2, 3, 2)
plt.imshow(gray_img, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

#Resize Image
plt.subplot(2,3,3)
plt.imshow(resized_img, cmap="gray")
plt.title("Resized Image")
plt.axis("off")


# --- Histogram of Original Image ---
plt.subplot(2, 3, 4)
plt.hist(gray_img.ravel(), 256, [0, 256], color='blue', alpha=1)
plt.title('Histogram (Original)')
plt.xlabel('Gray Level')
plt.ylabel('Number of Pixels')

# --- Equalized Image ---
plt.subplot(2, 3, 5)
plt.imshow(equalized, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

# --- Histogram of Equalized Image ---
plt.subplot(2, 3, 6)
plt.hist(equalized.ravel(), 256, [0, 256], color='green')
plt.title('Histogram (Equalized)')
plt.xlabel('Gray Level')
plt.ylabel('Number of Pixels')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()