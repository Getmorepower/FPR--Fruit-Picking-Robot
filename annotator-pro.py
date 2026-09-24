import os
import cv2

# 1. 配置参数与语义色彩 (BGR格式)
CLASSES = ["Pink_Lady_Apple", "Royal_Gala_Apple", "Granny_Smith"]
COLORS = [
    (203, 192, 255), # Pink_Lady_Apple: 粉色
    (0, 165, 255),   # Royal_Gala_Apple: 橙色
    (0, 255, 0)      # Granny_Smith_Apple: 绿色
]
TARGET_DIR = r'datasets/raw_images/Granny_Smith' # 切换文件夹

# 全局状态变量
current_idx = 0
images_list = []
current_boxes = []
original_img = None

def load_yolo_txt(txt_path):
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return boxes

def save_yolo_txt(txt_path, boxes):
    with open(txt_path, 'w') as f:
        for b in boxes:
            f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
    print(f"[保存成功] -> {os.path.basename(txt_path)}")

def draw_ui():
    """渲染 UI 与边界框"""
    global original_img, current_boxes
    img_display = original_img.copy()
    h, w, _ = img_display.shape
    
    # 绘制所有检测框
    for i, b in enumerate(current_boxes):
        cls_id, cx, cy, bw, bh = b
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        
        # 防御性色彩和标签提取
        color = COLORS[cls_id % len(COLORS)]
        label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"ID:{cls_id}"
        
        cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
        
        # 绘制带背景的文本标签，提升专业感
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_display, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(img_display, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 绘制顶部状态栏 (进度与文件名)
    progress_text = f" {current_idx + 1} / {len(images_list)} | {images_list[current_idx]} "
    cv2.putText(img_display, progress_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4) # 黑边
    cv2.putText(img_display, progress_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) # 白字
    
    # 绘制底部操作提示
    help_text = " [A/D]: Prev/Next | [Right-Click]: Delete Box | [S]: Save | [Q]: Quit "
    cv2.putText(img_display, help_text, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(img_display, help_text, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.imshow('Pro YOLO Annotator', img_display)

def mouse_event(event, x, y, flags, param):
    """鼠标回调：右键精准删框"""
    global current_boxes, original_img
    if event == cv2.EVENT_RBUTTONDOWN:
        h, w, _ = original_img.shape
        # 倒序遍历，确保优先删除最上层的框
        for i in range(len(current_boxes) - 1, -1, -1):
            cls_id, cx, cy, bw, bh = current_boxes[i]
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            
            # 判断点击位置是否在框内
            if x1 <= x <= x2 and y1 <= y <= y2:
                removed_box = current_boxes.pop(i)
                print(f"[-] 删除了一个多余的框: 类别 {removed_box[0]}")
                draw_ui()
                break # 每次点击只删一个

def start_engine(folder_path):
    global current_idx, images_list, current_boxes, original_img
    
    images_list = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not images_list:
        print("未检测到图片文件。")
        return

    cv2.namedWindow('Pro YOLO Annotator', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pro YOLO Annotator', 1024, 768)
    cv2.setMouseCallback('Pro YOLO Annotator', mouse_event) # 绑定鼠标事件

    while current_idx < len(images_list):
        img_name = images_list[current_idx]
        img_path = os.path.join(folder_path, img_name)
        txt_path = os.path.splitext(img_path)[0] + '.txt'

        original_img = cv2.imread(img_path)
        if original_img is None:
            current_idx += 1
            continue
            
        current_boxes = load_yolo_txt(txt_path)
        draw_ui()

        # 监听键盘事件
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('d') or key == ord('D'): # 下一张
            current_idx += 1
        elif key == ord('a') or key == ord('A'): # 上一张
            current_idx = max(0, current_idx - 1)
        elif key == ord('s') or key == ord('S'): # 保存当前修改
            save_yolo_txt(txt_path, current_boxes)
        elif key == ord('q') or key == ord('Q'): # 退出
            break

    cv2.destroyAllWindows()
    print("工程会话已结束。")

if __name__ == '__main__':
    print("启动 Pro 质检引擎...")
    start_engine(TARGET_DIR)