from networks.PGR_Net import PGR_Net
import torch
import time

def accurate_inference_timer(
    model, 
    input_shape=(1, 3, 224, 224),  # 显式定义输入尺寸（batch, channel, H, W）
    device="cuda",
    warmup=50,   # 预热次数
    repeat=200   # 正式计时次数
):
    # 1. 显式生成固定尺寸的输入张量（核心：尺寸可控且固定）
    input_tensor = torch.randn(*input_shape, device=device)
    
    # 2. 模型准备（规避eval模式/梯度的干扰）
    model = model.to(device).eval()
    with torch.no_grad():
        # 3. 预热：消除首次运行的CUDA初始化/算子编译开销
        for _ in range(warmup):
            _ = model(input_tensor)
        
        # 4. GPU同步：避免异步执行导致的计时偏差
        torch.cuda.synchronize()
        start = time.perf_counter()  # 高精度计时（比time.time()更准）
        
        # 5. 正式计时
        for _ in range(repeat):
            _ = model(input_tensor)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
    
    # 6. 计算统计值（精准且有意义）
    total_time = end - start
    avg_time_ms = (total_time / repeat) * 1000  # 平均单次时间（毫秒）
    fps = repeat / total_time                   # 每秒处理帧数
    
    # 输出时明确标注输入尺寸，避免歧义
    print(f"输入尺寸: {input_shape} | 平均推理时间: {avg_time_ms:.3f} ms")
    print(f"FPS: {fps:.2f} (每秒处理 {fps:.2f} 张图片)")
    return avg_time_ms, fps


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = PGR_Net(in_chan=3, base_chan=32, num_classes=9).to(device)

    avg_time, fps = accurate_inference_timer(
        net,
        input_shape=(1, 3, 224, 224),  # 明确指定输入尺寸
        device=device)

    try:
        from thop import profile
        input_tensor = torch.randn(1, 3, 224, 224).to(device)
        flops, params = profile(net, inputs=(input_tensor,))
        print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")
        print(f"Params (thop): {params / 1e6:.3f} M")
    except ImportError:
        print("请先安装 thop 库: pip install thop")

