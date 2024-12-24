# DCGAN
基于 DCGAN 实现动漫头像生成
# 项目名称

基于DCGAN的动漫头像生成网络。

## 环境要求

- Python 3.8
- 需要的包：
  - numpy
  - pandas
  - matplotlib

## 安装步骤

1. 安装所需的包：
   pip install -r requirements.txt
2. 安装所需的包：
   python main.py



## 注意事项

- **运行环境**：本项目使用 **macOS** 运行，支持 **Apple Silicon（M1/M2）** 芯片。
- **加速方式**：项目中利用 PyTorch 的 **Metal Performance Shaders（MPS）** 加速进行深度学习模型的训练与推理。在设备检测时，自动启用 `mps` 作为加速设备。
- **设备优先级**：如果运行设备不支持 MPS，代码将自动切换到 **CPU** 或 **CUDA**（若可用）。

