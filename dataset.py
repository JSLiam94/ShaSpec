import os
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_csv_from_folders(root_dir, output_csv):
    # 创建一个空列表来存储路径
    paths = []

    # 指定需要检查的目录
    target_dir = os.path.join(root_dir)

    # 使用os.walk遍历指定目录下的所有子文件夹
    for root, dirs, files in os.walk(target_dir):
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            paths.append(full_path)

    # 将路径列表转换为DataFrame
    df = pd.DataFrame(paths, columns=['path'])


    # 将DataFrame保存为CSV文件
    df.to_csv(output_csv, index=False)

# 设置根目录和输出CSV文件名
train_data = '/home/zhushenghao/data/ZSH/datasets/FeTS2024/train'
test_data = '/home/zhushenghao/data/ZSH/datasets/FeTS2024/test'
train_csv = './train.csv'
test_csv = './test.csv'

# 生成CSV文件
generate_csv_from_folders(train_data, train_csv)
generate_csv_from_folders(test_data, test_csv)




print("Train CSV file saved as 'train.csv'")
print("Validation CSV file saved as 'val.csv'")
