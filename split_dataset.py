import os
import shutil
import random
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# 配置参数
RAW_DIR = 'datasets/real_apple_dataset'
YOLO_DIR = 'datasets/real_apple_yolo_dataset'
CLASSES = ["Pink_Lady", "Royal_Gala", "Granny_Smith"]
TRAIN_RATIO = 0.8  # 80% 用于训练，20% 用于验证

def create_yolo_structure():
    """创建标准的 YOLO 文件夹结构"""
    directories = [
        f'{YOLO_DIR}/images/train',
        f'{YOLO_DIR}/images/val',
        f'{YOLO_DIR}/labels/train',
        f'{YOLO_DIR}/labels/val'
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    return directories

def verify_and_split_data():
    """验证文件成对存在，打乱并划分"""
    create_yolo_structure()
    
    total_train = 0
    total_val = 0

    for variety in CLASSES:
        variety_dir = os.path.join(RAW_DIR, variety)
        if not os.path.exists(variety_dir):
            continue
            
        # 收集所有成对的 (图片, txt)
        valid_pairs = []
        for file in os.listdir(variety_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(variety_dir, file)
                txt_path = os.path.join(variety_dir, os.path.splitext(file)[0] + '.txt')
                
                # 严谨的防御机制：只有当图片和 TXT 同时存在时，才纳入数据集
                if os.path.exists(txt_path):
                    valid_pairs.append((img_path, txt_path))
                else:
                    logging.warning(f"⚠️ 发现孤立的图片 (无标注): {file}")

        # 随机打乱以保证训练效果
        random.shuffle(valid_pairs)
        
        # 计算分割点
        split_index = int(len(valid_pairs) * TRAIN_RATIO)
        train_pairs = valid_pairs[:split_index]
        val_pairs = valid_pairs[split_index:]
        
        # 拷贝文件到目标目录 (使用 copy2 保留元数据，原 raw_images 作为备份)
        for img_src, txt_src in train_pairs:
            shutil.copy2(img_src, f'{YOLO_DIR}/images/train/')
            shutil.copy2(txt_src, f'{YOLO_DIR}/labels/train/')
            total_train += 1
            
        for img_src, txt_src in val_pairs:
            shutil.copy2(img_src, f'{YOLO_DIR}/images/val/')
            shutil.copy2(txt_src, f'{YOLO_DIR}/labels/val/')
            total_val += 1
            
        logging.info(f"✅ {variety} 处理完毕: {len(train_pairs)} 训练集, {len(val_pairs)} 验证集")

    logging.info(f"\n🎉 数据集构建成功！总计训练集: {total_train} 张，验证集: {total_val} 张。")

def generate_yaml():
    """自动生成 YOLO 需要的 dataset.yaml 配置文件"""
    # 获取绝对路径，YOLO 喜欢绝对路径以防止找不到文件
    abs_yolo_dir = os.path.abspath(YOLO_DIR)
    
    yaml_content = {
        'path': abs_yolo_dir,
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASSES),
        'names': {i: name for i, name in enumerate(CLASSES)}
    }
    
    yaml_path = os.path.join(abs_yolo_dir, 'real_dataset.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, sort_keys=False, allow_unicode=True)
        
    logging.info(f"📄 配置文件已生成: {yaml_path}")

if __name__ == "__main__":
    logging.info("启动数据清洗与重组引擎...")
    verify_and_split_data()
    generate_yaml()