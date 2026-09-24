import os
import cv2
import argparse
import colorsys
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import sys
import copy
import re

# AI 与追踪器初始化
try:
    from ultralytics import YOLO
    print("[AI] 正在初始化 YOLO 模型，初次运行可能需要下载基础权重...")
    AI_MODEL = YOLO('yolov8n.pt') 
    print("[AI] 成功加载 YOLOv8！按下 'M' 键即可一键智能预标注。")
except ImportError:
    AI_MODEL = None
    print("[AI] 提示: 未检测到 ultralytics，若需 AI 预标注请执行: pip install ultralytics")

def get_cv2_tracker():
    """获取兼容的 OpenCV 追踪器算法"""
    try: return cv2.TrackerCSRT_create()
    except Exception:
        try: return cv2.TrackerKCF_create()
        except Exception: 
            try: return cv2.TrackerMIL_create()
            except Exception: return None

# 1. 配置参数
# 移除配置，改为在运行时动态加载
CLASSES = []
COLORS = []

# 窗口尺寸配置
WORK_AREA_W = 1080  # 图像显示区固定宽度
WORK_AREA_H = 720   # 图像显示区固定高度
SIDEBAR_W = 320     # 右侧控制面板宽度

def generate_colors(num_classes):
    """根据类别数量动态生成高对比度的颜色 (BGR格式)"""
    colors = []
    if num_classes == 0:
        return colors
    for i in range(num_classes):
        hue = i / float(num_classes)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
        colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))
    return colors

# 全局状态变量
current_idx = 0
images_list = []
current_boxes = []
original_img = None
img_off_x = 0  # 图像在画布上的 X 偏移
img_off_y = 0  # 图像在画布上的 Y 偏移
mouse_x = -1   # 鼠标全局X坐标 (用于十字准星)
mouse_y = -1   # 鼠标全局Y坐标 (用于十字准星)
history_states = [] # 历史记录栈，用于实现撤销功能

# 画框相关的状态
drawing = False
ix, iy = -1, -1
current_draw_class = 0 # 默认当前画笔类别为 0 (Pink Lady)

# 核心读写逻辑
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

# 数据集管理引擎
def normalize_dataset(folder_path, prefix="image_"):
    """统一规范化命名新导入的图片，并同步它们的 .txt 标注文件"""
    print(f"[*] 正在检查并规范化数据集命名，规则: {prefix}xxxx.jpg ...")
    all_files = os.listdir(folder_path)
    images = [f for f in all_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # 1. 找出所有已经符合规范的图片，获取当前最大序号
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.(jpg|png|jpeg)$", re.IGNORECASE)
    max_idx = -1
    needs_rename = []
    
    for img in images:
        match = pattern.match(img)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx
        else:
            needs_rename.append(img)
            
    # 2. 对不符合规范的“新外来图片”进行重命名和预处理
    renamed_count = 0
    for img in needs_rename:
        max_idx += 1
        old_base, ext = os.path.splitext(img)
        new_img_name = f"{prefix}{max_idx:04d}{ext.lower()}"
        new_txt_name = f"{prefix}{max_idx:04d}.txt"
        
        old_img_path = os.path.join(folder_path, img)
        new_img_path = os.path.join(folder_path, new_img_name)
        old_txt_path = os.path.join(folder_path, old_base + '.txt')
        new_txt_path = os.path.join(folder_path, new_txt_name)
        
        # 重命名图片本体
        os.rename(old_img_path, new_img_path)
        
        # 同步迁移标注：如果你导入的新图片刚好自带同名的 txt 标注，一同改名；否则生成空 txt
        if os.path.exists(old_txt_path):
            os.rename(old_txt_path, new_txt_path)
        else:
            open(new_txt_path, 'a').close()
            
        renamed_count += 1
        
    # 3. 兜底策略：为已规范化但缺少 .txt 的老图片补齐空 txt，防止作为负样本时被 YOLO 抛弃
    for img in images:
        if img not in needs_rename:
            base = os.path.splitext(img)[0]
            txt_path = os.path.join(folder_path, base + '.txt')
            if not os.path.exists(txt_path):
                open(txt_path, 'a').close()
                
    if renamed_count > 0:
        print(f"[+] 数据集清洗完成！共吸收并重命名了 {renamed_count} 个新样本。")
    else:
        print("[*] 数据集命名很干净，无需整理。")

# 渲染引擎
def draw_ui(temp_pt=None):
    """
    渲染 UI 与所有检测框 (采用现代化深色面板分割设计)
    temp_pt: 如果传入坐标 (x, y)，则额外绘制一个鼠标拖拽中的预览框。
    """
    global original_img, current_boxes, current_draw_class, ix, iy, img_off_x, img_off_y, mouse_x, mouse_y
    
    if original_img is None: return
    img_display = original_img.copy()
    h, w, _ = img_display.shape

    # 创建深色现代背景固定尺寸画布 (暗黑模式)
    canvas_h = WORK_AREA_H
    canvas_w = WORK_AREA_W + SIDEBAR_W
    canvas = np.full((canvas_h, canvas_w, 3), (35, 35, 38), dtype=np.uint8)
    
    # 1. 绘制已保存的框
    for i, b in enumerate(current_boxes):
        cls_id, cx, cy, bw, bh = b
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        
        color = COLORS[cls_id % len(COLORS)]
        label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"ID:{cls_id}"
        
        cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_display, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(img_display, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 2. 绘制拖拽预览框
    if temp_pt is not None:
        color = COLORS[current_draw_class % len(COLORS)]
        cv2.rectangle(img_display, (ix, iy), temp_pt, color, 2)
        cv2.putText(img_display, "Drawing...", (ix, iy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 将渲染好的图像【居中】嵌入到画布图像显示区
    canvas[img_off_y:img_off_y+h, img_off_x:img_off_x+w] = img_display

    # 3. 绘制全屏十字准星 (Crosshair)，辅助精准对齐，仅限工作区
    if 0 <= mouse_x < WORK_AREA_W and 0 <= mouse_y < WORK_AREA_H:
        cv2.line(canvas, (mouse_x, 0), (mouse_x, WORK_AREA_H), (120, 200, 120), 1)
        cv2.line(canvas, (0, mouse_y), (WORK_AREA_W, mouse_y), (120, 200, 120), 1)

    # 侧边栏 UI
    # 面板左侧分割线
    ui_x = WORK_AREA_W
    cv2.line(canvas, (ui_x, 0), (ui_x, canvas_h), (60, 60, 65), 2)
    
    # 标题与当前文件信息
    cv2.putText(canvas, "Pro Annotator Studio", (ui_x + 20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.75, (250, 250, 250), 1)
    cv2.putText(canvas, "Data Operations", (ui_x + 20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.line(canvas, (ui_x + 20, 90), (ui_x + SIDEBAR_W - 20, 90), (60, 60, 65), 1)
    
    prog_text = f"Image: {current_idx + 1} / {len(images_list)}"
    cv2.putText(canvas, prog_text, (ui_x + 20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(canvas, f"File: {images_list[current_idx][:22]}", (ui_x + 20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    
    # 类别列表 (UI 可点击区)
    cv2.putText(canvas, "Label Classes (Click to Select):", (ui_x + 20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    for i, cls_name in enumerate(CLASSES):
        y_pos = 240 + i * 40
        color = COLORS[i % len(COLORS)]
        
        # 当前活动的类别高亮背景框
        if i == current_draw_class:
            cv2.rectangle(canvas, (ui_x + 10, y_pos - 25), (ui_x + SIDEBAR_W - 10, y_pos + 10), (80, 80, 85), -1)
            cv2.putText(canvas, ">", (ui_x + 15, y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        # 渲染颜色展示块
        cv2.rectangle(canvas, (ui_x + 40, y_pos - 18), (ui_x + 60, y_pos + 2), color, -1)
        cv2.rectangle(canvas, (ui_x + 40, y_pos - 18), (ui_x + 60, y_pos + 2), (200, 200, 200), 1)
        
        # 渲染类别名称
        cv2.putText(canvas, f"[{i+1}] {cls_name[:15]}", (ui_x + 75, y_pos - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)

    # 底部永久指引区
    inst_y = canvas_h - 150
    cv2.line(canvas, (ui_x + 20, inst_y), (ui_x + SIDEBAR_W - 20, inst_y), (60, 60, 65), 1)
    cv2.putText(canvas, "[L-Drag] Draw | [R-Click] Del", (ui_x + 20, inst_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(canvas, "[Z] Undo | [X] Clear | [C] Copy", (ui_x + 20, inst_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(canvas, "[T] Auto-Track | [M] AI Detect", (ui_x + 20, inst_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
    cv2.putText(canvas, "[A]/[D] Prev/Next (Auto-Save)", (ui_x + 20, inst_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(canvas, "[1-9] Switch Class | [Q] Quit", (ui_x + 20, inst_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow('Pro YOLO Annotator', canvas)

# 鼠标交互逻辑
def mouse_event(event, x, y, flags, param):
    """处理鼠标画框与删框事件"""
    global current_boxes, original_img, drawing, ix, iy, current_draw_class, img_off_x, img_off_y
    global mouse_x, mouse_y, history_states
    
    if original_img is None: return
    h, w, _ = original_img.shape
    
    mouse_x, mouse_y = x, y  # 全局记录鼠标位置用于准星刷新
    
    # 核心映射：将全局画布坐标系转换为【内部图片坐标系】
    img_x = x - img_off_x
    img_y = y - img_off_y
    
    # 验证鼠标是否在缩放后的真实图片范围内
    in_image_area = (0 <= img_x < w and 0 <= img_y < h)

    # 【功能1：左键按下】开始画框 / 面板交互
    if event == cv2.EVENT_LBUTTONDOWN:
        if in_image_area:
            drawing = True
            ix, iy = img_x, img_y
        elif x >= WORK_AREA_W:
            # UI面板点击识别：根据 Y 坐标计算是否点击到了某个类别的按钮
            if 215 <= y <= 215 + len(CLASSES) * 40:
                clicked_idx = (y - 215) // 40
                if 0 <= clicked_idx < len(CLASSES):
                    current_draw_class = clicked_idx
                    draw_ui()

    # 【功能2：鼠标移动】动态刷新预览框
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # 如果拖拽越界到了右侧面板，强制约束在图片边缘
            px, py = min(max(img_x, 0), w-1), min(max(img_y, 0), h-1)
            draw_ui((px, py)) 
        else:
            draw_ui() # 仅重绘，以更新十字准星的位置

    # 【功能3：左键抬起】结算坐标并写入内存
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            px, py = min(max(img_x, 0), w-1), min(max(img_y, 0), h-1)
            if abs(px - ix) > 5 and abs(py - iy) > 5:
                x_min, x_max = min(ix, px), max(ix, px)
                y_min, y_max = min(iy, py), max(iy, py)
                
                # 无损转换：即使图片经过缩放，归一化坐标也与原图完全一致
                cx = (x_min + x_max) / 2.0 / w
                cy = (y_min + y_max) / 2.0 / h
                bw = (x_max - x_min) / float(w)
                bh = (y_max - y_min) / float(h)
                
                current_boxes.append((current_draw_class, cx, cy, bw, bh))
                history_states.append(copy.deepcopy(current_boxes)) # 将新状态推入历史栈
                print(f"[+] 新增检测框: 类别 {current_draw_class}")
            draw_ui() # 刷新去掉预览框，显示正式框

    # 【功能4：右键按下】精准删除多余框
    elif event == cv2.EVENT_RBUTTONDOWN:
        if not in_image_area: return
        # 倒序遍历，确保优先删除图层最上面的框
        for i in range(len(current_boxes) - 1, -1, -1):
            cls_id, cx, cy, bw, bh = current_boxes[i]
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                removed_box = current_boxes.pop(i)
                history_states.append(copy.deepcopy(current_boxes)) # 将新状态推入历史栈
                print(f"[-] 删除检测框: 类别 {removed_box[0]}")
                draw_ui()
                break

# 主控制循环
def start_engine(folder_path, classes_file=None):
    global current_idx, images_list, current_boxes, original_img, current_draw_class
    global CLASSES, COLORS, img_off_x, img_off_y, history_states
    
    images_list = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not images_list:
        print("未检测到图片文件。")
        return
        
    # 动态加载类别与颜色配置
    if classes_file is None:
        classes_file = os.path.join(folder_path, 'classes.txt')
        
    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            CLASSES = [line.strip() for line in f.readlines() if line.strip()]
    else:
        print(f"[警告] 未找到 {classes_file}，将使用默认类别名称。")
        CLASSES = [f"Class_{i}" for i in range(10)]
        
    COLORS = generate_colors(max(10, len(CLASSES)))
    print(f"[配置] 成功加载 {len(CLASSES)} 个类别: {CLASSES}")

    # 更换为固态窗口（AUTOSIZE 意味着由程序画布完全接管窗口大小）
    cv2.namedWindow('Pro YOLO Annotator', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Pro YOLO Annotator', mouse_event)

    while current_idx < len(images_list):
        img_name = images_list[current_idx]
        img_path = os.path.join(folder_path, img_name)
        txt_path = os.path.splitext(img_path)[0] + '.txt'

        raw_img = cv2.imread(img_path)
        if raw_img is None:
            current_idx += 1
            continue
            
        # 核心自适应缩放算法
        raw_h, raw_w = raw_img.shape[:2]
        scale = min(WORK_AREA_W / raw_w, WORK_AREA_H / raw_h)
        new_w, new_h = int(raw_w * scale), int(raw_h * scale)
        original_img = cv2.resize(raw_img, (new_w, new_h))
        
        # 计算居中所需的黑边偏移量
        img_off_x = (WORK_AREA_W - new_w) // 2
        img_off_y = (WORK_AREA_H - new_h) // 2
        
        current_boxes = load_yolo_txt(txt_path)
        history_states = [copy.deepcopy(current_boxes)] # 初始化当前图片的历史记录
        draw_ui()

        # 监听键盘事件
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('d') or key == ord('D'):
            save_yolo_txt(txt_path, current_boxes) # 无感自动保存
            current_idx += 1
        elif key == ord('a') or key == ord('A'):
            save_yolo_txt(txt_path, current_boxes) # 无感自动保存
            current_idx = max(0, current_idx - 1)
        elif key == ord('s') or key == ord('S'):
            save_yolo_txt(txt_path, current_boxes)
        elif key == ord('z') or key == ord('Z') or key == 26: # 26通常是 CV2 下 Ctrl+Z 的 ASCII 码
            if len(history_states) > 1:
                history_states.pop()
                current_boxes = copy.deepcopy(history_states[-1])
                draw_ui()
                print("[!] 已撤销上一步操作")
        elif key == ord('x') or key == ord('X'):
            if current_boxes:
                current_boxes.clear()
                history_states.append(copy.deepcopy(current_boxes))
                draw_ui()
                print("[-] 已清空当前图片所有检测框")
        elif key == ord('c') or key == ord('C'):
            if current_idx > 0:
                prev_img_path = os.path.join(folder_path, images_list[current_idx - 1])
                prev_txt_path = os.path.splitext(prev_img_path)[0] + '.txt'
                prev_boxes = load_yolo_txt(prev_txt_path)
                if prev_boxes:
                    current_boxes.extend(prev_boxes)
                    history_states.append(copy.deepcopy(current_boxes))
                    draw_ui()
                    print("[+] 已成功从上一张图片复制检测框")
        elif key == ord('t') or key == ord('T'):
            if current_idx > 0:
                print("[*] 正在使用 OpenCV 追踪上一张图的目标...")
                prev_img_path = os.path.join(folder_path, images_list[current_idx - 1])
                prev_txt_path = os.path.splitext(prev_img_path)[0] + '.txt'
                prev_boxes = load_yolo_txt(prev_txt_path)
                
                raw_prev = cv2.imread(prev_img_path)
                if raw_prev is not None and prev_boxes:
                    ph, pw = raw_prev.shape[:2]
                    tracked_count = 0
                    for b in prev_boxes:
                        cls_id, cx, cy, bw, bh = b
                        x1, y1 = int((cx - bw/2)*pw), int((cy - bh/2)*ph)
                        bw_px, bh_px = int(bw*pw), int(bh*ph)
                        
                        tracker = get_cv2_tracker()
                        if tracker is None:
                            print("[!] 当前 OpenCV 版本不支持 Tracking，请尝试更新 opencv-python")
                            break
                        try:
                            x1, y1 = max(0, x1), max(0, y1)
                            bw_px = min(pw - x1, bw_px)
                            bh_px = min(ph - y1, bh_px)
                            
                            tracker.init(raw_prev, (x1, y1, bw_px, bh_px))
                            success, box = tracker.update(raw_img)
                            
                            if success:
                                tx, ty, tw, th = box
                                ncx, ncy = (tx + tw/2)/raw_w, (ty + th/2)/raw_h
                                nbw, nbh = tw/raw_w, th/raw_h
                                current_boxes.append((cls_id, ncx, ncy, nbw, nbh))
                                tracked_count += 1
                        except Exception as e:
                            print(f"[!] 目标追踪异常: {e}")
                    
                    if tracked_count > 0:
                        history_states.append(copy.deepcopy(current_boxes))
                        draw_ui()
                        print(f"[+] 成功跨帧追踪了 {tracked_count} 个目标！")
                    else:
                        print("[-] 追踪失败，目标可能移动过快或发生了严重遮挡。")
        elif key == ord('m') or key == ord('M'):
            if AI_MODEL is not None:
                print("[*] 正在运行 YOLOv8 智能分析...")
                results = AI_MODEL(raw_img, conf=0.3, verbose=False)
                new_boxes_count = 0
                for r in results:
                    for box in r.boxes:
                        xywhn = box.xywhn[0].tolist()
                        # 默认将 AI 检出的目标全部分配为你当前选中的标签类型（以防AI的字典和你自己的字典不匹配）
                        current_boxes.append((current_draw_class, float(xywhn[0]), float(xywhn[1]), float(xywhn[2]), float(xywhn[3])))
                        new_boxes_count += 1
                if new_boxes_count > 0:
                    history_states.append(copy.deepcopy(current_boxes))
                    draw_ui()
                    print(f"[+] AI 预标注完成，生成了 {new_boxes_count} 个候选框！(按 Z 撤销或右键删除多余)")
                else:
                    print("[-] AI 未检测到明显目标。")
            else:
                print("[!] 未安装 ultralytics 库。若需使用此功能，请在终端执行: pip install ultralytics")
        # 切换当前画笔的品种类别
        elif ord('1') <= key <= ord('9'):
            cls_idx = key - ord('1')
            if cls_idx < len(CLASSES):
                current_draw_class = cls_idx
                draw_ui()
        elif key == ord('q') or key == ord('Q'):
            break

    cv2.destroyAllWindows()
    print("工程会话已结束。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pro YOLO Annotator - 通用目标检测标注工具")
    parser.add_argument('-d', '--dir', type=str, default=None, help="包含图像和YOLO标注txt的目录路径 (不指定则弹出选择框)")
    parser.add_argument('-c', '--classes', type=str, default=None, help="classes.txt 类别文件路径 (默认读取图像目录下的 classes.txt)")
    parser.add_argument('-p', '--prefix', type=str, default=None, help="可选: 自动重命名新添加的混乱图片 (例如设为: frame_)")
    args = parser.parse_args()

    target_dir = args.dir
    target_prefix = args.prefix
    # 如果命令行没有提供目录，则弹出图形化选择框
    if not target_dir:
        root = tk.Tk()
        root.withdraw()  # 隐藏不需要的 tkinter 主窗口
        target_dir = filedialog.askdirectory(title="请选择包含图像和YOLO标注的文件夹")
        if not target_dir:
            print("未选择任何文件夹，程序已安全退出。")
            sys.exit(0)

        # 可视化模式下，如果在命令行没有传入 prefix，通过弹窗询问用户
        if target_prefix is None:
            if messagebox.askyesno("数据集智能整理", "是否需要自动识别并重命名文件夹中新导入的（命名混乱的）图片？\n\n这会让新数据按顺序自动编号，并补齐空 txt 文件。"):
                target_prefix = simpledialog.askstring("设置前缀", "请输入统一命名的新前缀（例如 frame_, image_）:", initialvalue="image_")

    # 如果用户启用了重命名清洗规则（无论通过命令行还是 GUI），先对文件夹进行“洗牌”
    if target_prefix:
        normalize_dataset(target_dir, target_prefix)

    print("启动 Pro 质检引擎...")
    start_engine(target_dir, args.classes)