#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# 封锁 CPU 多线程，算力让给 ROS2 通信层
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import threading
from collections import deque
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from ultralytics import YOLO
import std_srvs.srv

from servo_controller_msgs.msg import ServosPosition, ServoPosition

sys.path.append('/home/ubuntu/ros2_ws/src/driver/kinematics/')
try:
    import kinematics.transform as transform
    from kinematics.inverse_kinematics import get_ik
    HAS_IK = True
except ImportError as e:
    HAS_IK = False
    print(f"找不到 IK 引擎，请确认路径: {e}")

class TargetKalmanFilter:
    """针对相机坐标系下 3D 目标点 (X, Y, Z) 的线性卡尔曼滤波器"""
    def __init__(self):
        # 状态向量: [x, y, z, vx, vy, vz]^T, 观测向量: [x, y, z]^T
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)

        self.dt = 0.033  # 约 30 FPS 的采样间隔
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)

        # 过程噪声与测量噪声协方差矩阵 (针对 2~5mm 深度噪声优化)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-3
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        
        self.initialized = False

    def update(self, measurement):
        meas = np.array([[np.float32(measurement[0])],
                         [np.float32(measurement[1])],
                         [np.float32(measurement[2])]], dtype=np.float32)
        
        if not self.initialized:
            self.kf.statePost = np.array([
                [meas[0][0]], [meas[1][0]], [meas[2][0]],
                [0.0], [0.0], [0.0]
            ], dtype=np.float32)
            self.initialized = True
            return measurement

        self.kf.predict()
        estimate = self.kf.correct(meas)
        return float(estimate[0][0]), float(estimate[1][0]), float(estimate[2][0])

    def reset(self):
        self.initialized = False

class AppleVisualServoingNode(Node):
    def __init__(self):
        super().__init__('apple_visual_servoing_node')
        
        self.STATE_INIT_HOME = 0        
        self.STATE_IDLE_SEARCH = 1      
        self.STATE_YAW_ALIGN = 2        
        self.STATE_SNAP_WAIT = 3        
        self.STATE_FINAL_STRIKE = 4     
        self.STATE_HUMAN_CONFIRM = 5    
        
        self.current_state = self.STATE_INIT_HOME
        self.depth_assist_enabled = False
        
        # 添加底盘控制总开关 (False: 锁死底盘, True: 允许移动)
        self.chassis_enabled = False
        self.hud_log_msg = "System Booting... Waiting for commands."
        
        self.apple_classes = {
            0: "Pink Lady",
            1: "Royal Gala",
            2: "Granny Smith"
        }
        self.mission_sequence = [] 
        self.current_mission_idx = 0
        
        # 引入“闪电缓存”，只存最原始的数据包，不消耗算力解码
        self.latest_rgb_msg = None
        self.latest_depth_msg = None
        
        pt_path = "yolov8n_pruned_30percent.engine" 
        self.get_logger().info("正在加载 YOLO 视觉中枢...")
        self.model = YOLO(pt_path, task='detect')
        self.model.task = 'detect'
        
        self.confidence_threshold = 0.50
        self.target_bboxes = []  
        self.depth_image = None
        
        self.fx, self.fy = 358.892548, 358.892548
        self.cx, self.cy = 319.224731, 178.908173
        
        self.offset_x = 0.01
        self.offset_y = 0.02
        self.offset_z = 0.16
        self.yaw_scale = 1.2
        
        self.sweet_zone_min = 0.22  
        self.sweet_zone_max = 0.25 
        
        self.ready_pulses = {1: 500, 2: 750, 3: 50, 4: 150, 5: 500, 10: 50} 
        self.strike_arm_dict = {}

        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.camera_subscription = self.create_subscription(RosImage, '/depth_cam/rgb/image_raw', self.camera_callback, qos)
        self.depth_subscription = self.create_subscription(RosImage, '/depth_cam/depth/image_raw', self.depth_callback, qos)
        
        self.servo_pub = self.create_publisher(ServosPosition, '/servo_controller', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10) 
        
        self.fsm_timer = self.create_timer(0.1, self.state_machine_heartbeat)
        # 初始化 3D 目标坐标卡尔曼滤波器
        self.target_kf = TargetKalmanFilter()
        
        cv2.namedWindow("Apple Harvesting HUD", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Apple Harvesting HUD", 1024, 600)
        cv2.setMouseCallback("Apple Harvesting HUD", self.mouse_click_event)
        self.get_logger().info("HUD 面板已就绪！")
        
        # 以下增强：AWB 锁定 + FPS + 调试发布器
        self._lock_camera_awb()
        self._fps_history = deque(maxlen=30)
        self._last_fps = 0.0
        self._home_cmd_sent = False
        self._home_time = 0.0
        self.debug_pub = self.create_publisher(RosImage, '/apple_vision/debug_image', 10)
        self._hotkey_help_visible = False

    def update_log(self, msg):
        self.get_logger().info(msg)
        self.hud_log_msg = msg

    # 鼠标交互回调引擎
    def mouse_click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if 700 < x < 950:
                if 50 < y < 110: self.start_mission([0])  
                elif 130 < y < 190: self.start_mission([1])  
                elif 210 < y < 270: self.start_mission([2])  
                elif 290 < y < 350: self.start_mission([1, 2, 0])  
                elif 390 < y < 450: self.toggle_depth_assist() 
                elif 490 < y < 570: self.confirm_take_apple()  

    def publish_servo_cmd(self, duration, id_pulse_dict):
        msg = ServosPosition()
        msg.duration = float(duration)
        msg.position_unit = 'pulse'
        for servo_id, pulse in id_pulse_dict.items():
            pos = ServoPosition()
            pos.id = int(servo_id)
            pos.position = float(pulse)
            msg.position.append(pos)
        self.servo_pub.publish(msg)

    def move_chassis(self, linear_x):
        twist = Twist()
        twist.linear.x = float(linear_x)
        self.cmd_vel_pub.publish(twist)
        
    def stop_chassis(self):
        self.cmd_vel_pub.publish(Twist())

    def start_mission(self, sequence):
        self.target_kf.reset()  # 重置滤波历史
        self.mission_sequence = sequence
        self.current_mission_idx = 0
        self.current_state = self.STATE_INIT_HOME
        seq_names = [self.apple_classes[i] for i in sequence]
        self.update_log(f"MISSION ACCEPTED: {seq_names}")

    def confirm_take_apple(self):
        if self.current_state == self.STATE_HUMAN_CONFIRM:
            self.target_kf.reset()  # 重置滤波历史
            self.update_log("APPLE TAKEN. RETURNING TO BASE...")
            self.current_mission_idx += 1
            if self.current_mission_idx >= len(self.mission_sequence):
                self.update_log("ALL MISSIONS ACCOMPLISHED! STANDBY.")
            else:
                next_apple = self.apple_classes[self.mission_sequence[self.current_mission_idx]]
                self.update_log(f"NEXT TARGET: {next_apple}")
            self.current_state = self.STATE_INIT_HOME

    def toggle_depth_assist(self):
        self.depth_assist_enabled = not self.depth_assist_enabled
        if self.depth_assist_enabled:
            self.update_log("🛡️ DEPTH ASSIST SHIELD: ON")
        else:
            self.update_log("🛡️ DEPTH ASSIST SHIELD: OFF")

    def state_machine_heartbeat(self):
        if self.current_state == self.STATE_INIT_HOME:
            if not self._home_cmd_sent:
                self.publish_servo_cmd(duration=1.5, id_pulse_dict=self.ready_pulses)
                self._home_cmd_sent = True
                self._home_time = time.time()
            elif time.time() - self._home_time > 1.5:
                self._home_cmd_sent = False
                if self.current_mission_idx < len(self.mission_sequence):
                    target_name = self.apple_classes[self.mission_sequence[self.current_mission_idx]]
                    self.update_log(f"SEARCHING FOR: [{target_name}] ...")
                    self.current_state = self.STATE_IDLE_SEARCH
        else:
            self._home_cmd_sent = False

    # 增强视觉方法

    @staticmethod
    def _simplest_cb(roi: np.ndarray, percent: float = 1.0) -> np.ndarray:
        if roi.size == 0:
            return roi
        out = roi.astype(np.float32)
        for c in range(3):
            ch = roi[..., c]
            lo = float(np.percentile(ch, percent))
            hi = float(np.percentile(ch, 100 - percent))
            if hi - lo < 1.0:
                continue
            out[..., c] = np.clip((ch.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
        return out.astype(np.uint8)

    def _clahe(self, roi: np.ndarray) -> np.ndarray:
        if roi.size == 0:
            return roi
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        planes = list(cv2.split(lab))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        planes[0] = clahe.apply(planes[0])
        return cv2.cvtColor(cv2.merge(planes), cv2.COLOR_LAB2BGR)

    def _verify_color(self, roi: np.ndarray, cls_id: int) -> int:
        """
        Multi-method color verification: SimplestCB + CLAHE + HSV + RGB ratios.
        Apple colors:
          Granny Smith (2): solid green throughout
          Royal Gala   (1): red/orange skin with yellow patches
          Pink Lady    (0): pink-red, minimal yellow/green
        """
        if roi.size == 0:
            return cls_id
        roi_cb = self._simplest_cb(roi, 1.0)
        roi_eq = self._clahe(roi_cb)
        hsv = cv2.cvtColor(roi_eq, cv2.COLOR_BGR2HSV)
        mask_red    = cv2.inRange(hsv, np.array([0, 40, 40]),   np.array([10, 255, 255]))
        mask_yellow = cv2.inRange(hsv, np.array([11, 40, 40]),  np.array([30, 255, 255]))
        mask_green  = cv2.inRange(hsv, np.array([40, 40, 40]),  np.array([85, 255, 255]))
        red_px = cv2.countNonZero(mask_red)
        yellow_px = cv2.countNonZero(mask_yellow)
        green_px = cv2.countNonZero(mask_green)
        total = red_px + yellow_px + green_px + 1
        mean_b, mean_g, mean_r = cv2.mean(roi_eq)[:3]
        ratio_rg = mean_r / max(mean_g, 0.01)
        if green_px / total > 0.4 or (ratio_rg < 0.9 and mean_g > mean_r):
            return 2
        yellow_ratio = yellow_px / total
        if yellow_ratio > 0.15:
            return 1
        elif red_px / total > 0.3 and yellow_ratio < 0.05:
            return 0
        return cls_id

    def _lock_camera_awb(self):
        camera_ns = "/depth_cam"
        svc_name = f"{camera_ns}/set_color_auto_white_balance"
        cli = self.create_client(std_srvs.srv.SetBool, svc_name)
        if cli.wait_for_service(timeout_sec=1.0):
            req = std_srvs.srv.SetBool.Request()
            req.data = False
            future = cli.call_async(req)
            future.add_done_callback(
                lambda f: self.get_logger().info(
                    "AWB locked via service" if f.result().success else "AWB lock service returned false"))
            self.get_logger().info("Locking AWB via Orbbec service...")
        else:
            param_names = [
                f"{camera_ns}/enable_color_auto_white_balance",
                f"{camera_ns}/rgb_camera.enable_auto_white_balance",
            ]
            locked = False
            for pname in param_names:
                try:
                    self.set_parameters([rclpy.parameter.Parameter(
                        pname, rclpy.parameter.Parameter.Type.BOOL, False)])
                    self.get_logger().info(f"AWB locked via param: {pname}")
                    locked = True
                    break
                except Exception:
                    continue
            if not locked:
                self.get_logger().warn(
                    "Could not lock AWB — camera driver may not support it. "
                    "Fallback: Simplest Color Balance + RGB ratios active.")

    # 防止相机驱动拥堵
    def camera_callback(self, msg):
        self.latest_rgb_msg = msg

    def depth_callback(self, msg):
        self.latest_depth_msg = msg

    # Game Loop 处理图像
    
    def core_game_loop(self):
        if self.latest_rgb_msg is None:
            cv2.waitKey(1)
            return

        try:
            # 1. 取出图像并销毁缓存，防止重复处理
            rgb_msg = self.latest_rgb_msg
            self.latest_rgb_msg = None 
            
            # 延迟解码：只消耗 CPU 解码我们真正要处理的这一帧！
            raw_data = np.frombuffer(rgb_msg.data, dtype=np.uint8)
            cv_image = np.ndarray(shape=(rgb_msg.height, rgb_msg.width, 3), dtype=np.uint8, buffer=raw_data, strides=(rgb_msg.step, 3, 1))
            if hasattr(rgb_msg, 'encoding') and 'rgb' in rgb_msg.encoding.lower():
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            cv_image = np.ascontiguousarray(cv_image.copy())

            # FPS 计数
            now = time.time()
            self._fps_history.append(now)
            if len(self._fps_history) > 1:
                span = self._fps_history[-1] - self._fps_history[0]
                self._last_fps = (len(self._fps_history) - 1) / span if span > 0 else 0
            
            # 如果深度图就绪，也进行延迟解码
            if self.latest_depth_msg is not None:
                d_msg = self.latest_depth_msg
                self.latest_depth_msg = None
                raw_depth = np.frombuffer(d_msg.data, dtype=np.uint16)
                self.depth_image = np.ndarray(shape=(d_msg.height, d_msg.width), dtype=np.uint16, buffer=raw_depth, strides=(d_msg.step, 2)).copy()
            
            self.target_bboxes = [] 
            
            # 深度护盾接管逻辑
            if self.current_state in [self.STATE_IDLE_SEARCH, self.STATE_YAW_ALIGN] and self.depth_assist_enabled:
                if self.depth_image is not None:
                    mask = cv2.inRange(self.depth_image, 150, 400)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_c = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(largest_c) > 1000: 
                            x, y, w, h = cv2.boundingRect(largest_c)
                            self.target_bboxes = [[x, y, x+w, y+h]]
                            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (255, 0, 0), 3)
                            cv2.putText(cv_image, "🛡️ DEPTH LOCK", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # YOLO 与 HSV 双层语义视觉逻辑
            # 只有当没有被深度护盾强行接管时，才运行 YOLO (节约算力)
            if len(self.target_bboxes) == 0:
                results = self.model(cv_image, verbose=False)
                # 修复死锁：将 SNAP_WAIT 和 FINAL_STRIKE 也加入锁定白名单，确保目光至死不渝！
                active_states = [self.STATE_IDLE_SEARCH, self.STATE_YAW_ALIGN, self.STATE_SNAP_WAIT, self.STATE_FINAL_STRIKE]
                mission_active = (self.current_state in active_states and self.current_mission_idx < len(self.mission_sequence))
                current_target_class = self.mission_sequence[self.current_mission_idx] if mission_active else -1

                for box in results[0].boxes:
                    detected_class = int(box.cls[0])
                    if float(box.conf[0]) >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # 多层颜色验证：SimplestCB + CLAHE + HSV + RGB 比值
                        if detected_class in [0, 1]:
                            roi = cv_image[y1:y2, x1:x2]
                            if roi.size > 0:
                                detected_class = self._verify_color(roi, detected_class)
                                self.get_logger().info(
                                    f"色彩扫描 -> 判定为: {self.apple_classes[detected_class]}")
                        
                        # 锁定框渲染
                        if mission_active and detected_class == current_target_class:
                            self.target_bboxes.append([x1, y1, x2, y2])
                            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(cv_image, f"LOCK: {self.apple_classes[detected_class]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (200, 200, 200), 2)
                            cv2.putText(cv_image, f"{self.apple_classes[detected_class]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # 状态机推进
            if self.current_state == self.STATE_IDLE_SEARCH and len(self.target_bboxes) > 0:
                self.process_first_look()
            elif self.current_state == self.STATE_SNAP_WAIT and len(self.target_bboxes) > 0:
                self.process_second_look()

            # 渲染 HUD
            dashboard = np.zeros((600, 1024, 3), dtype=np.uint8)
            dashboard[:] = (40, 40, 40) 
            
            cam_resized = cv2.resize(cv_image, (640, 480))
            dashboard[40:520, 20:660] = cam_resized
            cv2.rectangle(dashboard, (20, 40), (660, 520), (200, 200, 200), 2)
            cv2.putText(dashboard, f"LIVE: ROBOT VISION  |  FPS: {self._last_fps:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            
            btn_w, btn_h = 250, 60
            cv2.rectangle(dashboard, (700, 50), (700+btn_w, 50+btn_h), (80, 80, 220), -1)
            cv2.putText(dashboard, "1. PINK LADY", (720, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.rectangle(dashboard, (700, 130), (700+btn_w, 130+btn_h), (0, 200, 255), -1)
            cv2.putText(dashboard, "2. ROYAL GALA", (720, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            cv2.rectangle(dashboard, (700, 210), (700+btn_w, 210+btn_h), (100, 200, 100), -1)
            cv2.putText(dashboard, "3. GRANNY SMITH", (715, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            cv2.rectangle(dashboard, (700, 290), (700+btn_w, 290+btn_h), (200, 100, 200), -1)
            cv2.putText(dashboard, "ULTIMATE: ALL 3", (720, 328), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            shield_color = (0, 200, 0) if self.depth_assist_enabled else (100, 100, 100)
            cv2.rectangle(dashboard, (700, 390), (700+btn_w, 390+btn_h), shield_color, 2)
            cv2.putText(dashboard, f"SHIELD: {'ON' if self.depth_assist_enabled else 'OFF'}", (720, 428), cv2.FONT_HERSHEY_SIMPLEX, 0.7, shield_color, 2)
            
            confirm_color = (0, 100, 255) if self.current_state == self.STATE_HUMAN_CONFIRM else (80, 80, 80)
            cv2.rectangle(dashboard, (700, 490), (700+btn_w, 490+btn_h+20), confirm_color, -1)
            cv2.putText(dashboard, "CONFIRM TAKEN", (730, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if self._hotkey_help_visible:
                overlay = dashboard.copy()
                cv2.rectangle(overlay, (10, 200), (630, 440), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.85, dashboard, 0.15, 0, dashboard)
                helps = [
                    ("1/2/3", "Single apple mission"),
                    ("4", "Ultimate (all 3)"),
                    ("5", "Toggle depth shield"),
                    ("6", "Confirm apple taken"),
                    ("C", "Toggle chassis enable"),
                    ("H", "Toggle this help"),
                ]
                for i, (key, desc) in enumerate(helps):
                    y = 230 + i * 30
                    cv2.putText(dashboard, f"{key}  {desc}", (30, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.putText(dashboard, f"STATUS: {self.hud_log_msg}", (20, 560),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow("Apple Harvesting HUD", dashboard)
            key = cv2.waitKey(1) & 0xFF
            self._handle_hotkey(key)
            
        except Exception as e:
            self.get_logger().error(f"Main Loop Error: {e}")

    def _handle_hotkey(self, key):
        if key == ord('1'):
            self.start_mission([0])
        elif key == ord('2'):
            self.start_mission([1])
        elif key == ord('3'):
            self.start_mission([2])
        elif key == ord('4'):
            self.start_mission([1, 2, 0])
        elif key == ord('5'):
            self.toggle_depth_assist()
        elif key == ord('6'):
            self.confirm_take_apple()
        elif key == ord('c'):
            self.chassis_enabled = not self.chassis_enabled
            self.update_log(f"🚗 Chassis {'ENABLED' if self.chassis_enabled else 'DISABLED'}")
        elif key == ord('h'):
            self._hotkey_help_visible = not self._hotkey_help_visible

    
    # 解析及抓取动作模块
    
    def parse_coordinates(self):
        if self.depth_image is None or not self.target_bboxes: return None
        x1, y1, x2, y2 = self.target_bboxes[0]
        depth_crop = self.depth_image[y1:y2, x1:x2]
        non_zero_depths = depth_crop[depth_crop > 0]
        if len(non_zero_depths) < 50: return None
        
        # 1. 获取原始中值深度 (Raw Depth)
        raw_z_meters = np.median(non_zero_depths) / 1000.0
        box_center_x, box_center_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        # 2. 深度非线性漂移补偿
        safe_distance = 0.25 
        if raw_z_meters > safe_distance:
            z_compensation = 0.15 * ((raw_z_meters - safe_distance) ** 2)
            z_meters = raw_z_meters + z_compensation
        else:
            z_meters = raw_z_meters

        # 3. 计算修正后的 X 和 Y
        x_meters = (box_center_x - self.cx) * z_meters / self.fx
        y_meters = (box_center_y - self.cy) * z_meters / self.fy
        
        # 4. 边缘畸变补偿
        offset_ratio = abs(box_center_x - self.cx) / self.cx 
        compensation_x = (offset_ratio ** 2) * 0.05  
        if x_meters > 0:
            x_meters += compensation_x 
        else:
            x_meters -= compensation_x 

        # 5. 卡尔曼滤波平滑输出 (消除跳变与抖动)
        smooth_x, smooth_y, smooth_z = self.target_kf.update((x_meters, y_meters, z_meters))
        return smooth_x, smooth_y, smooth_z

    def process_first_look(self):
        coords = self.parse_coordinates()
        if coords is None:
            self.stop_chassis()
            return
            
        cx, cy, cz = coords
        if cz > self.sweet_zone_max:
            self.move_chassis(linear_x=0.05)
            return 
        elif cz < self.sweet_zone_min:
            self.move_chassis(linear_x=-0.05) 
            return 
            
        self.stop_chassis() 
        self.current_state = self.STATE_YAW_ALIGN
        target_x, target_y, target_z = cz + self.offset_x, -cx + self.offset_y, -cy + self.offset_z
        safe_x, safe_y, safe_z = 0.20, target_y * (0.20 / target_x), 0.10
        
        res = get_ik([safe_x, safe_y, safe_z], 0.0, [-180, 180])
        if not res:
            self.current_state = self.STATE_IDLE_SEARCH
            return
            
        ik_pulses = transform.angle2pulse(res[0][0])
        raw_yaw = float(np.array(ik_pulses[0]).flatten()[0])
        self.locked_yaw = max(100.0, min(900.0, 500 + (raw_yaw - 500) * self.yaw_scale))
        self.publish_servo_cmd(duration=1.0, id_pulse_dict={1: self.locked_yaw, 10: 50})
        self.snap_timer = self.create_timer(1.5, self.transit_to_snap)

    def transit_to_snap(self):
        if hasattr(self, 'snap_timer') and self.snap_timer is not None:
            self.destroy_timer(self.snap_timer)
            self.snap_timer = None
        if self.current_state == self.STATE_YAW_ALIGN:
            self.current_state = self.STATE_SNAP_WAIT
            
    def process_second_look(self):
        coords = self.parse_coordinates()
        if coords is None: return
        cx, cy, cz = coords
        self.current_state = self.STATE_FINAL_STRIKE
        
        target_x, target_y, target_z = cz + self.offset_x, -cx + self.offset_y, -cy + self.offset_z
        # 抓取时的俯仰角定在 45 度
        res = get_ik([target_x, target_y, target_z], 45.0, [40.0, 60.0])
        
        if not res:
            self.update_log("❌ IK FAILED! TARGET UNREACHABLE.")
            self.current_state = self.STATE_INIT_HOME
            return
            
        ik_pulses = transform.angle2pulse(res[0][0])
        flat_pulses = ik_pulses[0] if isinstance(ik_pulses[0], list) or type(ik_pulses[0]).__name__ == 'ndarray' else ik_pulses
        
        self.strike_arm_dict = {}
        for i in range(len(flat_pulses)):
            self.strike_arm_dict[i+1] = float(np.array(flat_pulses[i]).flatten()[0])
            
        self.strike_arm_dict[1] = self.locked_yaw if hasattr(self, 'locked_yaw') else 500
        self.strike_arm_dict[10] = 50 # 爪子张开
        
        # 核心修改：把目标坐标 (target_x, target_y, target_z) 传给出击线程，用于计算回收轨迹！
        threading.Thread(target=self.execute_blind_strike, args=(target_x, target_y, target_z)).start()

    # 核心修改：接收坐标，执行三段式动力学回收
    def execute_blind_strike(self, tx, ty, tz):
        try:
            # 动作 1：45度俯冲下探 
            self.update_log("STRIKING (45-DEGREE ANGLE)...")
            self.publish_servo_cmd(duration=3.0, id_pulse_dict=self.strike_arm_dict) 
            time.sleep(3.5) 
            
            # 动作 2：咬合夹紧 
            self.update_log("GRASPING APPLE...")
            self.publish_servo_cmd(duration=1.5, id_pulse_dict={10: 680}) 
            time.sleep(2.0)
            
            # 动作 3：斜向托举 (Clearance 验证抓取) 
            self.update_log("LIFTING GENTLY (VERIFYING GRASP)...")
            lift_x, lift_y, lift_z = tx - 0.04, ty, tz + 0.08
            res_lift = get_ik([lift_x, lift_y, lift_z], 60.0, [45.0, 80.0])
            
            if res_lift:
                lift_pulses = transform.angle2pulse(res_lift[0][0])
                flat_lift = lift_pulses[0] if isinstance(lift_pulses[0], list) or type(lift_pulses[0]).__name__ == 'ndarray' else lift_pulses
                lift_pose = {i+1: float(np.array(flat_lift[i]).flatten()[0]) for i in range(len(flat_lift))}
                lift_pose[1] = self.strike_arm_dict.get(1, 500)
                lift_pose[10] = 680  # 保持抓紧
                
                self.publish_servo_cmd(duration=1.5, id_pulse_dict=lift_pose)
                time.sleep(2.0)

            # 动作 3.5：原路逆向放回原位
            self.update_log("RETURNING APPLE TO ORIGINAL POSITION...")
            put_back_pose = self.strike_arm_dict.copy()
            put_back_pose[10] = 680  # 保持闭合状态下放
            self.publish_servo_cmd(duration=2.0, id_pulse_dict=put_back_pose)
            time.sleep(2.3)

            # 动作 3.6：松开机械爪
            self.update_log("RELEASING APPLE...")
            self.publish_servo_cmd(duration=1.0, id_pulse_dict={10: 50})
            time.sleep(1.2)

            # 动作 4：空爪贴胸安全收拢
            self.update_log("RETRACTING TO SECURE HOME POSE...")
            res_carry = get_ik([0.18, ty * 0.5, 0.15], 0.0, [-30.0, 30.0])
            
            if res_carry:
                carry_pulses = transform.angle2pulse(res_carry[0][0])
                flat_carry = carry_pulses[0] if isinstance(carry_pulses[0], list) or type(carry_pulses[0]).__name__ == 'ndarray' else carry_pulses
                carry_pose = {i+1: float(np.array(flat_carry[i]).flatten()[0]) for i in range(len(flat_carry))}
                carry_pose[1] = self.strike_arm_dict.get(1, 500)
                carry_pose[10] = 50  # 空爪收拢
                
                self.publish_servo_cmd(duration=2.0, id_pulse_dict=carry_pose)
                time.sleep(2.5)
            
            self.update_log("MISSION CYCLE COMPLETE! CLICK [CONFIRM TAKEN].")
            self.current_state = self.STATE_HUMAN_CONFIRM
            
        except Exception as e:
            self.get_logger().error(f"Strike Error: {e}")
            self.current_state = self.STATE_INIT_HOME

# ROS2 心跳执行流
def main(args=None):
    rclpy.init(args=args)
    node = AppleVisualServoingNode()
    try:
        while rclpy.ok():
            # 1. 允许底层的相机通信和定时器快速通过
            rclpy.spin_once(node, timeout_sec=0.01)
            # 2. 按照节奏，处理最新的一帧图像，并绘制 HUD
            node.core_game_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
