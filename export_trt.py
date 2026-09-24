import sys
from unittest.mock import MagicMock
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')


# 动态伪造 matplotlib，防止触发 NumPy 2.0 崩溃
logging.info("🛡️ 正在启动动态模块拦截，绕过环境冲突...")
mock_module = MagicMock()
# 将所有可能引发崩溃的画图模块全部重定向到我们的“假库”上
sys.modules['matplotlib'] = mock_module
sys.modules['matplotlib.pyplot'] = mock_module
sys.modules['matplotlib.colors'] = mock_module
sys.modules['matplotlib.rcsetup'] = mock_module
sys.modules['matplotlib.scale'] = mock_module
sys.modules['matplotlib.ticker'] = mock_module
sys.modules['matplotlib.transforms'] = mock_module
sys.modules['matplotlib._path'] = mock_module

# 拦截完成，导入 YOLO
from ultralytics import YOLO

def export_to_tensorrt():
    logging.info("🚀 唤醒 Jetson GPU，开始编译专属 TensorRT 引擎...")
    
    # 确保模型名字完全一致
    model = YOLO('yolov8n_pruned_30percent.pt')  
    
    model.export(
        format='engine',
        device='0',         
        half=True,          # FP16 半精度量化
        simplify=True,      
        workspace=2         
    )
    
    logging.info("✅ 硬件加速引擎编译完成！")

if __name__ == '__main__':
    export_to_tensorrt()