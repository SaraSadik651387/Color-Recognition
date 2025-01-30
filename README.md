# 🎨 Color Detection using OpenCV

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-orange?style=flat-square&logo=googlecolab)

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
<img width="319" alt="Image" src="https://github.com/user-attachments/assets/721e46e1-801d-4219-aeab-50cc9f3932f7" />

### 🎯 **Processed Output**
<img width="575" alt="Image" src="https://github.com/user-attachments/assets/ee4240cc-c04f-4be1-901d-8aafc2f6ff08" />


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

# Load the image
image_path = "apple.jpg"  # Change this to the name of the uploaded image in Google Colab
image = cv2.imread(image_path)

# Convert the image to HSV format
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges for Red, Green, and Blue in HSV
colors = {
    "Red": ([0, 120, 70], [10, 255, 255]),  # Lower and upper HSV bounds for red
    "Green": ([36, 100, 100], [86, 255, 255]),  # Lower and upper HSV bounds for green
    "Blue": ([94, 80, 2], [126, 255, 255])  # Lower and upper HSV bounds for blue
}

# Process each color
for color_name, (lower, upper) in colors.items():
    lower = np.array(lower)  # Convert lower bound to numpy array
    upper = np.array(upper)  # Convert upper bound to numpy array

    # Create a mask for the color
    mask = cv2.inRange(hsv_image, lower, upper)

    # Find the contours of the detected color regions
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Ignore small regions
            x, y, w, h = cv2.boundingRect(contour)  # Get bounding box of the contour
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle

            # Improve text visibility with shadow effect
            cv2.putText(image, color_name, (x+2, y-8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)  # Black shadow
            cv2.putText(image, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)  # White text

# Display the image with detected color regions and labels
cv2_imshow(image)

```
## 🛠 Setup & Installation  

### 📌 1. Clone the Repository  
📥 **If you want to run the project locally on your computer:**  
Open **Terminal** or **Command Prompt**, then copy and paste this command:

**Command:**  
`git clone https://github.com/SaraSadik651387/Color-Recognition.git`  
`cd Color-Recognition-OpenCV`  

---

### 📌 2. Install Dependencies  
📦 **Make sure you have all required libraries installed.**  

**Command:**  
`pip install opencv-python numpy`  

🔹 **Libraries:**  
- **`opencv-python`** → OpenCV library for image processing.  
- **`numpy`** → Numerical operations library for handling images.  

---

### 📌 3. Run the Script Locally  
📸 **To execute the color detection script:**  

**Command:**  
`python Color_Recognition.ipynb`  

---

### 📌 4. Run the Project in Google Colab  
📌 **If you prefer running the project in Google Colab (without installing anything on your device), follow these steps:**  

**Steps in Google Colab:**  
1. Upload the image you want to test (Go to **Files → Upload**).  
2. Run all the cells step by step.  
3. Watch the results directly in your browser.  

---

### 📌 5. Upload Your Own Images  
📸 **To test different images, upload any image to Google Colab:**  

**Steps:**  
1. Click the Files icon in Colab.  
2. Select "Upload" and choose an image from your device.  
3. Replace `image_path` in the code with your uploaded image name:    

🔹 **Notes:**  
- Run the code and see the results!  

