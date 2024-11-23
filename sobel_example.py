import os
from skimage import io, filters, img_as_ubyte
import matplotlib.pyplot as plt

folder_path = 'C:/Users/李晓璇/Desktop/Semester 3/CS6123/project_3/spriters/Forage'

output_folder = os.path.join('C:/Users/李晓璇/Desktop/Semester 3/CS6123/project_3/scikit-image/Forage', 'output_edges')
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        file_path = os.path.join(folder_path, file_name)

        image = io.imread(file_path, as_gray=True)

        sobel_edges = filters.sobel(image)

        sobel_edges_uint8 = img_as_ubyte(sobel_edges)

        output_path = os.path.join(output_folder, f'sobel_{file_name}')
        io.imsave(output_path, sobel_edges_uint8)
        print(f"Processed and saved: {output_path}")

print("All images processed.")
