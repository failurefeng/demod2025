import torch
from flash_attn import flash_attn_func

# 生成测试数据（必须将 q, k, v 合并为单个张量）
batch_size = 2
seq_len = 10
num_heads = 8
d_model = 256
q = torch.randn(batch_size, seq_len, num_heads, d_model).cuda().to(torch.float16)
k = torch.randn(batch_size, seq_len, num_heads, d_model).cuda().to(torch.float16)
v = torch.randn(batch_size, seq_len, num_heads, d_model).cuda().to(torch.float16)

# 调用flash_attn_func（注意输入格式）
output = flash_attn_func(
    q, k, v,
    causal=True  # 启用因果注意力
)
print(output.shape)  # 应为 (2, 10, 256)

