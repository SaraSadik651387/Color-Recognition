# 🎨 Color Detection using OpenCV

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-orange?style=flat-square&logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square)

---

## 🌟 **Overview**
This project detects and labels **Red 🔴, Green 🟢, and Blue 🔵 colors** in an image using **OpenCV and Python**.  
It applies **computer vision techniques** to:
- Identify color regions in images
- Draw bounding boxes 📦 around detected objects
- Label detected colors with text ✍️

🔥 This is a beginner-friendly **image processing project** that helps in understanding **color segmentation using HSV (Hue, Saturation, Value)** color space.

---

## 📸 **Demo Output**
Below is an example of how the script detects colors in an image:

### 🖼 **Input Image**
<img src="https://upload.wikimedia.org/wikipedia/commons/2/23/Traffic_lights.jpg" width="450" height="300">

### 🎯 **Processed Output**
<img src="https://www.publicdomainpictures.net/pictures/10000/velka/1-1210009435EGmE.jpg" width="450" height="300">

---

## 🔧 **Technologies Used**
| Technology | Description |
|------------|------------|
| ![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python) | Programming Language |
| ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv) | Computer Vision Library |
| ![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-orange?style=flat-square&logo=googlecolab) | Jupyter Notebook Environment |

---

## 🚀 **How It Works**
1️⃣ **Load an Image**: The program reads an image using OpenCV.  
2️⃣ **Convert to HSV Format**: The image is converted to the HSV color space for better color filtering.  
3️⃣ **Apply Color Masking**: Specific color ranges are used to detect **Red, Green, and Blue**.  
4️⃣ **Detect & Label Colors**: The script:
   - Identifies colored regions
   - Draws bounding boxes around detected colors
   - Adds text labels on top of detected areas  

---

## 📝 **Code Snippet**
Here’s a simplified version of the **main logic**:

```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load Image
image = cv2.imread("apple.jpg")
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define Color Ranges (HSV)
colors = {
    "Red": ([0, 120, 70], [10, 255, 255]),
    "Green": ([36, 100, 100], [86, 255, 255]),
    "Blue": ([94, 80, 2], [126, 255, 255])
}

# Detect Colors
for color_name, (lower, upper) in colors.items():
    mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

cv2_imshow(image)
