import torch
import os
from scene.scene_shader import SceneShader
from scene.colmap_loader import qvec2rotmat
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


def calculate_h(qandt):

    diff_sum_all = []
    with torch.no_grad():  # 禁用梯度计算，因为在渲染过程中不需要梯度信息


        scene = SceneShader(dataset, height, width, qandt,
                            cam_intrinsic_first, cam_intrinsic_second, load_iteration=args.iteration)  # 创建一个 Scene 对象，用于处理场景的渲染

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]  # 根据数据集的背景设置，定义背景颜色
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 将背景颜色转换为 PyTorch 张量，同时将其移到 GPU 上

        # 调用 render_set 函数渲染训练数据集
        #  render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        for idx in range(len(scene.getTrainCameras())):
            rendering = render(scene.getTrainCameras()[idx], gaussians, pipeline, background)["render"]  # 这里执行的就是上面解析过的render的代码了~

            diff_sum = torch.abs(rendering - query_image_tensor).sum()
            diff_sum_all.append(diff_sum.item())

            render_path = "output/output_rendering.png"
            render_path1 = "output/output_gt.png"
            torchvision.utils.save_image(rendering, render_path)
            torchvision.utils.save_image(query_image_tensor, render_path1)


    # print(f"两幅图像每个像素之间的差值之和为: {diff_sum.item()}")  # 使用 item() 方法获取张量中的标量值

    return diff_sum_all



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

    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)  # 创建一个 GaussianModel 对象，用于处理高斯模型
    gaussians.load_ply(os.path.join(args.model_path,
                                    "point_cloud",
                                    "iteration_30000",    #################
                                    "point_cloud.ply"))

    query_image_list = "data/db/drjohnson/results_temp.txt"
    with open(query_image_list, "r") as f:
        for data in f.read().rstrip().split("\n"):
            data = data.split()
            name = data[0]

            print('cur_image: ', name)

            qvec_s0, tvec_s0 = np.split(np.array(data[1:], float), [4])

            # last_folder = name.split('/')[0]   ## 7scenes
            # image_name_in_seq = name.split('/')[1]  ## 7scenes
            
            base_image_folder = "/root/autodl-tmp/Hierarchical-Localization/datasets/db/drjohnson"

            # query_image_path = base_image_folder + "/" + last_folder + "/" + image_name_in_seq   ## 7scenes

            query_image_path = base_image_folder + "/" + name    ## playroom

            # 加载图像并进行转换
            query_image = Image.open(query_image_path)  # 使用 PIL 库加载图像

            target_width = int(query_image.size[0] / 1)
            target_height = int(query_image.size[1] / 1)

            transform = transforms.Compose([
                transforms.Resize((target_height, target_width), transforms.InterpolationMode.BICUBIC),  # 等比例缩小
                transforms.CenterCrop((target_height, target_width)),  # 剪裁中心部分
                transforms.ToTensor()  # 将图像转换为 torch 张量
            ])

            width, height = query_image.size

            cam_intrinsic_first = 1035.49659905  ######
            cam_intrinsic_second = 1034.97186374

            query_image_tensor = transform(query_image)  # 将图像转换为 torch 张量
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            query_image_tensor = query_image_tensor.to(device)

            OPEN = []
            CLOSE = []
            isv = {}
            dist = {}

            s0 = [float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6]), float(data[7])]

            s0_h = calculate_h([s0])[0]



