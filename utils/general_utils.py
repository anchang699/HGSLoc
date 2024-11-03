import torch
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


# 提取协方差矩阵的上三角部分，并将其展平为一维张量
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    # 提取协方差矩阵的上三角部分，并将其展平为一维张量
    return strip_lowerdiag(sym)


def build_rotation(r):
    # 计算四元数 r 的模长
    norm = torch.sqrt(r[:, 0]*r[:, 0] + r[:, 1]*r[:, 1] + r[:, 2]*r[:, 2] + r[:, 3]*r[:, 3])
    # 将四元数除以其模长进行标准化
    # 考虑 norm 是一个形状为 (n,) 的张量，r 是一个形状为 (n, 4) 的张量。
    # 在这种情况下，norm[:, None] 将 norm 转变为形状为 (n, 1) 的张量，以便与 r 的形状对齐进行元素级的除法运算。
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


# 构建一个尺度和旋转组合的变换矩阵
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")  # s.shape[0]可能为数量
    R = build_rotation(r)
    # 将 s 中的尺度参数分别赋值给 L 的对角线元素
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

# 该函数的作用是在执行期间重定向标准输出（sys.stdout）到一个新的类 F 的实例。
# 类 F 在写入时会检查是否需要在每行结尾处添加时间戳
def safe_state(silent):
    old_f = sys.stdout  # 将当前的标准输出流（sys.stdout）保存在 old_f 变量中
    class F:
        def __init__(self, silent):
            self.silent = silent  # 用于表示是否为静默状态

        def write(self, x):
            if not self.silent:
                # 如果输入的内容 x 以换行符结尾且非静默状态，则在每行末尾添加一个时间戳，并输出到标准输出。
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()  # 用于刷新标准输出流

    sys.stdout = F(silent)  # 创建 F 类的实例并将其赋值给 sys.stdout，从而重定向标准输出到新的类实例

    # 设置随机种子以确保结果的可重复性。
    random.seed(0)  # 设置 Python 内置的随机数生成器的种子为 0。
    np.random.seed(0)  # 设置 NumPy 库的随机数生成器的种子为 0。
    torch.manual_seed(0)  # 设置 PyTorch 库的随机数生成器的种子为 0。
    torch.cuda.set_device(torch.device("cuda:0"))
