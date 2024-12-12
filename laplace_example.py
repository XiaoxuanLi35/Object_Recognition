import os
from skimage import io, filters, img_as_ubyte
import numpy as np

# 设置输入和输出文件夹路径
folder_path = 'C:/Users/myname/Desktop/Semester 3/CS6123/project_3/spriters/Forage'
output_folder = os.path.join('C:/Users/myname/Desktop/Semester 3/CS6123/project_3/scikit-image/Forage', 'laplace_edges')
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 读取图片路径
        file_path = os.path.join(folder_path, file_name)
        
        # 读取图像并转换为灰度图
        image = io.imread(file_path, as_gray=True)
        
        # 应用Laplace滤波器
        laplace_edges = filters.laplace(image)
        
        # 将结果归一化到[0,1]范围
        laplace_edges = np.abs(laplace_edges)
        laplace_edges = (laplace_edges - laplace_edges.min()) / (laplace_edges.max() - laplace_edges.min())
        
        # 转换为uint8格式(0-255)
        laplace_edges_uint8 = img_as_ubyte(laplace_edges)
        
        # 保存结果
        output_path = os.path.join(output_folder, f'laplace_{file_name}')
        io.imsave(output_path, laplace_edges_uint8)
        print(f"Processed and saved: {output_path}")

print("All images processed.")
