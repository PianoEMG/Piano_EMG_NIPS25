import os
import json
import random

def split_dataset(data_dir, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    # 1. 获取文件夹中的所有文件名
    files = os.listdir(data_dir)
    # files = [f for f in files if os.path.isfile(os.path.join(data_dir, f))] 
    files = [f for f in files if os.path.isfile(os.path.join(data_dir, f)) and '_p1_' in f]
    # if len(files) == 7000:
    if len(files) == 350:
        print(f"Totally {len(files)} performances.")
    else:
        print(f"Error! Totally {len(files)} performances.")
        return

    # 2. 打乱文件顺序
    random.shuffle(files)

    # 3. 计算每个子集的大小
<<<<<<< Updated upstream
    train_size = 5000
    test_size = 1500
    val_size = len(files) - train_size - test_size  # 剩下的作为 val
=======
    # train_size = 6000
    # test_size = 800
    # val_size = len(files) - train_size - test_size  
    train_size = 300
    test_size = 40
    val_size = len(files) - train_size - test_size
>>>>>>> Stashed changes

    # 4. 切分数据集
    train_files = files[:train_size]
    test_files = files[train_size:train_size + test_size]
    val_files = files[train_size + test_size:]

    # 5. 创建数据集字典
    dataset = {
        'train': train_files,
        'test': test_files,
        'val': val_files
    }

    # 6. 保存为JSON文件
    json_file = 'temp_p1.json'
    with open(json_file, 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f'Dataset split and saved to {json_file}')

# 使用该函数
<<<<<<< Updated upstream
data_dir = 'C:\\Users\\ruofa\\Desktop\\Piano_Dataset\\emg_data'  # 这里替换为你的数据文件夹路径
=======
data_dir = 'C:/Users/ruofa/Desktop/Piano_Dataset/keystroke_data/'  # 这里替换为你的数据文件夹路径
>>>>>>> Stashed changes
split_dataset(data_dir)
