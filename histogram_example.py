import cv2
import numpy as np
from pathlib import Path
import os
import locale

# Set encoding to support Chinese in the path
locale.setlocale(locale.LC_ALL, '')

# Target folders
input_path = Path(r"C:\Users\name\Desktop\Semester 3\CS6123\project_3\spriters\Forage")
output_path = Path(r"C:\Users\name\Desktop\Semester 3\CS6123\project_3\scikit-image\Forage\color_histogram")
output_path.mkdir(exist_ok=True)

def extract_color_histogram(image_path):
   try:
       # Read image in binary mode to handle folders with Chinese
       with open(image_path, 'rb') as f:
           img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
           img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
           
       if img is None:
           print(f"Cannot read image: {image_path}")
           return None
           
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
       hist = cv2.normalize(hist, hist).flatten()
       return hist
   except Exception as e:
       print(f"Error processing image {image_path}: {str(e)}")
       return None

# Process all images
for img_path in input_path.glob('*.png'):
   print(f"Processing: {img_path.name}")
   hist = extract_color_histogram(img_path)
   if hist is not None:
       save_path = output_path / f"{img_path.stem}_histogram.npy"
       np.save(str(save_path), hist)
