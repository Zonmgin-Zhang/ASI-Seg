import os

# 指定第一个文件夹的路径
folder_path = '/data/zzm/SurgicalSAM-main/data/endovis_2018/train/40/sam_features_h'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".npy"):
        # 构建新文件名（移除下划线）
        new_name = filename.split("npy.npy")[0]
        new_filename = new_name + '.npy'
        # 重命名文件
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
