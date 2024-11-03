import torch
import os
from scene.scene_shader import SceneShader
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import heapq


class Node:
    def __init__(self, m, fa, g, h):
        self.m = m
        self.fa = fa
        self.g = g
        self.h = h
        self.id = 0

    def __lt__(self, other):
        # return (self.g + self.h) < (other.g + other.h)

        if (self.g + self.h) != (other.g + other.h):
            return (self.g + self.h) < (other.g + other.h)
        return self.m < other.m

# def calculate_h_single(m):
#     qvec = np.array(m[:4])
#     tvec = np.array(m[4:])
#
#     with torch.no_grad():  # 禁用梯度计算，因为在渲染过程中不需要梯度信息
#
#         gaussians = GaussianModel(dataset.sh_degree)  # 创建一个 GaussianModel 对象，用于处理高斯模型
#
#         scene = SceneShader(dataset, gaussians, height, width, qvec, tvec,
#                             cam_intrinsic_first, cam_intrinsic_second, load_iteration=args.iteration)  # 创建一个 Scene 对象，用于处理场景的渲染
#
#         bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]  # 根据数据集的背景设置，定义背景颜色
#         background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 将背景颜色转换为 PyTorch 张量，同时将其移到 GPU 上
#
#         # 调用 render_set 函数渲染训练数据集
#         #  render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
#         rendering = render(scene.getTrainCameras()[0], gaussians, pipeline, background)["render"]  # 这里执行的就是上面解析过的render的代码了~
#
#         # torchvision.utils.save_image(rendering, os.path.join(dataset.model_path, "0000.png"))
#
#
#     diff_sum = torch.abs(rendering - refer_image_tensor).sum()
#     # print(f"两幅图像每个像素之间的差值之和为: {diff_sum.item()}")  # 使用 item() 方法获取张量中的标量值
#
#     return diff_sum.item()


def calculate_h(qandt):

    diff_sum_all = []
    with torch.no_grad():  # 禁用梯度计算，因为在渲染过程中不需要梯度信息

        gaussians = GaussianModel(dataset.sh_degree)  # 创建一个 GaussianModel 对象，用于处理高斯模型

        scene = SceneShader(dataset, gaussians, height, width, qandt,
                            cam_intrinsic_first, cam_intrinsic_second, load_iteration=args.iteration)  # 创建一个 Scene 对象，用于处理场景的渲染

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]  # 根据数据集的背景设置，定义背景颜色
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 将背景颜色转换为 PyTorch 张量，同时将其移到 GPU 上

        # 调用 render_set 函数渲染训练数据集
        #  render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        for idx in range(len(scene.getTrainCameras())):
            rendering = render(scene.getTrainCameras()[idx], gaussians, pipeline, background)["render"]  # 这里执行的就是上面解析过的render的代码了~

            diff_sum = torch.abs(rendering - refer_image_tensor).sum()
            diff_sum_all.append(diff_sum.item())
    # print(f"两幅图像每个像素之间的差值之和为: {diff_sum.item()}")  # 使用 item() 方法获取张量中的标量值

    return diff_sum_all


def judge_dist_and_deg(q1, t1, q2, t2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    # 计算四元数的内积
    dot_product = np.dot(q1, q2)

    # # 将内积值限制在 [-1, 1] 范围内
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # 计算旋转角度
    theta_rad = 2 * np.arccos(dot_product)
    theta_deg = np.degrees(theta_rad)

    # 判断旋转角度是否小于2度
    is_less_than_deg = theta_deg < 10

    # 计算平移向量之间的欧几里得距离
    distance = np.linalg.norm(t2 - t1)

    # 判断位置偏移是否小于1米
    is_less_than_m = distance < 1

    return is_less_than_m and is_less_than_deg


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    start = time.time()

    height = 598
    width = 799
    cam_intrinsic_first = 658.40980263
    cam_intrinsic_second = 750.32784033

    dataset = model.extract(args)

    # 倒数第二张
    # qvec = np.array([-0.06496904, 0.6046065, -0.59156484, -0.52941575])
    # tvec = np.array([-1.05781704, -1.43584378, -1.10699879])

    # 倒数第三张
    # qvec = np.array([-0.11860639,  0.64060242, -0.59415012, -0.47174855])
    # tvec = np.array([-0.81685361, -0.98742696, -0.36579443])


    refer_image_path = "D:/PyCharmProject/comment_3DGS/data/playroom_resized/images/DSC05803.jpg"
    transform = transforms.Compose([
        transforms.ToTensor()  # 将图像转换为 torch 张量
    ])
    # 加载图像并进行转换
    refer_image = Image.open(refer_image_path)  # 使用 PIL 库加载图像
    refer_image_tensor = transform(refer_image)  # 将图像转换为 torch 张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refer_image_tensor = refer_image_tensor.to(device)


    # -----------------------------------------------------------------------------------

    s0 = [-0.06531239,  0.60187956, -0.58962926, -0.53461447, -1.05588659, -1.42316032, -1.09607316]
    g = [-0.11898412,  0.63823423, -0.59229789, -0.47716146, -0.8159356, -0.97542444, -0.35526241]
    qvec_s0 = np.array(s0[:4])
    tvec_s0 = np.array(s0[4:])
    qvec_g = np.array(g[:4])
    tvec_g = np.array(g[4:])

    OPEN = []
    CLOSE = []
    isv = {}
    dist = {}


    node = Node(s0, -1, 0, calculate_h([s0])[0])
    isv[tuple(node.m)] = True
    dist[tuple(node.m)] = 0
    heapq.heappush(OPEN, node)

    while OPEN:
        top = OPEN[0]
        heapq.heappop(OPEN)
        top.id = len(CLOSE)
        CLOSE.append(top)
        print(top.h)

        # qvec_top = np.array(top.m[:4])
        # tvec_top = np.array(top.m[4:])

        if top.h < 22000:
            print("success!")
            print(top.m)
            break

        # for pos_item, value_item in [(0, 0.01), (0, -0.01), (1, 0.01), (1, -0.01), (2, 0.01), (2, -0.01), (3, 0.01), (3, -0.01),
        #                              (4, 0.1), (4, -0.1), (5, 0.1), (5, -0.1), (6, 0.1), (6, -0.1)]:
        #     topm_copy = top.m[:]
        #     topm_copy[pos_item] += value_item
        #     qvec_cur = np.array(topm_copy[:4])
        #     tvec_cur = np.array(topm_copy[4:])
        #
        #     if judge_dist_and_deg(qvec_s0, tvec_s0, qvec_cur, tvec_cur):
        #
        #         child = Node(topm_copy[:], top.id, top.g + 1, 0)
        #
        #         child.h = calculate_h(child.m)
        #
        #         if tuple(child.m) not in isv or top.g + 1 < dist[tuple(child.m)]:
        #             isv[tuple(child.m)] = True
        #             dist[tuple(child.m)] = top.g + 1
        #             heapq.heappush(OPEN, child)

        pose_changed_lst = []

        for pos_item, value_item in [(0, 0.01), (0, -0.01), (1, 0.01), (1, -0.01), (2, 0.01), (2, -0.01), (3, 0.01), (3, -0.01),
                                     (4, 0.1), (4, -0.1), (5, 0.1), (5, -0.1), (6, 0.1), (6, -0.1)]:
            topm_copy = top.m[:]
            topm_copy[pos_item] += value_item
            qvec_cur = np.array(topm_copy[:4])
            tvec_cur = np.array(topm_copy[4:])

            if judge_dist_and_deg(qvec_s0, tvec_s0, qvec_cur, tvec_cur):

                pose_changed_lst.append(topm_copy)

        diff_sum_values = calculate_h(pose_changed_lst)

        for pose_idx in range(len(pose_changed_lst)):
            child = Node(pose_changed_lst[pose_idx][:], top.id, top.g + 1, diff_sum_values[pose_idx])
            if tuple(child.m) not in isv or top.g + 1 < dist[tuple(child.m)]:
                isv[tuple(child.m)] = True
                dist[tuple(child.m)] = top.g + 1
                heapq.heappush(OPEN, child)




    # with torch.no_grad():  # 禁用梯度计算，因为在渲染过程中不需要梯度信息
    #
    #     gaussians = GaussianModel(dataset.sh_degree)  # 创建一个 GaussianModel 对象，用于处理高斯模型
    #
    #     scene = SceneShader(dataset, gaussians, height, width, qvec, tvec,
    #                         cam_intrinsic_first, cam_intrinsic_second, load_iteration=args.iteration)  # 创建一个 Scene 对象，用于处理场景的渲染
    #
    #     bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]  # 根据数据集的背景设置，定义背景颜色
    #     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 将背景颜色转换为 PyTorch 张量，同时将其移到 GPU 上
    #
    #     # 调用 render_set 函数渲染训练数据集
    #     #  render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
    #     rendering = render(scene.getTrainCameras()[0], gaussians, pipeline, background)["render"]  # 这里执行的就是上面解析过的render的代码了~
    #
    #     # torchvision.utils.save_image(rendering, os.path.join(dataset.model_path, "0000.png"))
    #
    #
    # # 计算两幅图像每个像素之间的差值之和
    # diff_sum = torch.abs(rendering - image_tensor).sum()
    # print(f"两幅图像每个像素之间的差值之和为: {diff_sum.item()}")  # 使用 item() 方法获取张量中的标量值
    #
    #
    end = time.time()

    print('time:', end-start)
