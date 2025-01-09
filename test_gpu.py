import torch
print("hello")
print(torch.__version__)                # 检查 PyTorch 版本
print(torch.cuda.is_available())        # 检查是否支持 GPU
print(torch.cuda.get_device_name(0))    # 查看 GPU 名称