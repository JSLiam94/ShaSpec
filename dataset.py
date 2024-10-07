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
root_directory = '/root/autodl-tmp/BraTS2024/train'
output_csv_file = './output.csv'

# 生成CSV文件
generate_csv_from_folders(root_directory, output_csv_file)

# 读取CSV文件
df = pd.read_csv(output_csv_file)

# 划分数据集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 保存结果
train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)

print("Train CSV file saved as 'train.csv'")
print("Validation CSV file saved as 'val.csv'")
