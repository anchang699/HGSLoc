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

            # render_path = "output_rendering.png"
            # render_path1 = "output_gt.png"
            # torchvision.utils.save_image(rendering, render_path)
            # torchvision.utils.save_image(query_image_tensor, render_path1)


    # print(f"两幅图像每个像素之间的差值之和为: {diff_sum.item()}")  # 使用 item() 方法获取张量中的标量值

    return diff_sum_all


def judge_dist_and_deg(q1, t1, q2, t2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    # # 计算四元数的内积
    # dot_product = np.dot(q1, q2)
    #
    # # # 将内积值限制在 [-1, 1] 范围内
    # dot_product = np.clip(dot_product, -1.0, 1.0)
    #
    # # 计算旋转角度
    # theta_rad = 2 * np.arccos(dot_product)
    # theta_deg = np.degrees(theta_rad)
    #
    # # 判断旋转角度是否小于2度
    # is_less_than_deg = theta_deg < 5
    #
    # # 计算平移向量之间的欧几里得距离
    # distance = np.linalg.norm(t2 - t1)
    #
    # # 判断位置偏移是否小于1米
    # is_less_than_m = distance < 0.5


    R1 = qvec2rotmat(q1)
    R2 = qvec2rotmat(q2)

    e_t = np.linalg.norm(-R1.T @ t1 + R2.T @ t2, axis=0)
    cos = np.clip((np.trace(np.dot(R1.T, R2)) - 1) / 2, -1.0, 1.0)
    e_R = np.rad2deg(np.abs(np.arccos(cos)))

    is_less_than_m = e_t <= 0.5
    is_less_than_deg = e_R < 5

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

    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)  # 创建一个 GaussianModel 对象，用于处理高斯模型
    gaussians.load_ply(os.path.join(args.model_path,
                                    "point_cloud",
                                    "iteration_30000",
                                    "point_cloud.ply"))

    query_image_list = "data/7scenes/stairs/results_sparse.txt"
    with open(query_image_list, "r") as f:
        for data in f.read().rstrip().split("\n"):
            data = data.split()
            name = data[0]

            print('cur_image: ', name)

            qvec_s0, tvec_s0 = np.split(np.array(data[1:], float), [4])

            last_folder = name.split('/')[0]
            image_name_in_seq = name.split('/')[1]
            base_image_folder = "/root/autodl-tmp/Hierarchical-Localization/datasets/7scenes/stairs"

            query_image_path = base_image_folder + "/" + last_folder + "/" + image_name_in_seq

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

            cam_intrinsic_first = 525  ######
            cam_intrinsic_second = 320

            query_image_tensor = transform(query_image)  # 将图像转换为 torch 张量
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            query_image_tensor = query_image_tensor.to(device)

            OPEN = []
            CLOSE = []
            isv = {}
            dist = {}

            s0 = [float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6]), float(data[7])]

            s0_h = calculate_h([s0])[0]

            node = Node(s0, -1, 0, s0_h)
            isv[tuple(node.m)] = True
            dist[tuple(node.m)] = 0
            heapq.heappush(OPEN, node)

            num_node = 0
            best_h = s0_h + 1
            best_m = []
            best_num_node = -1

            while OPEN:
                top = OPEN[0]
                heapq.heappop(OPEN)
                top.id = len(CLOSE)
                CLOSE.append(top)
                # print(top.h)
                num_node += 1
                if top.h < best_h:
                    best_h = top.h
                    best_m = top.m
                    # print('cur_best_no: ', num_node)
                    best_num_node = num_node
                if num_node >= 400:
                    print(best_m)
                    # print('best_num_node: ', best_num_node)
                    break

                # if top.h < s0_h*0.2:  # 37160
                #     # print("success!")
                #     best_m = top.m
                #     print(top.m)
                #     break

                pose_changed_lst = []

                for pos_item, value_item in [(0, 1e-5), (0, -1e-5), (1, 1e-5), (1, -1e-5), (2, 1e-5), (2, -1e-5), (3, 1e-5), (3, -1e-5),
                                             (4, 1e-5), (4, -1e-5), (5, 1e-5), (5, -1e-5), (6, 1e-5), (6, -1e-5)

                                             # (0, 0.0001), (0, -0.0001), (1, 0.0001), (1, -0.0001), (2, 0.0001), (2, -0.0001), (3, 0.0001), (3, -0.0001),
                                             # (4, 0.0001), (4, -0.0001), (5, 0.0001), (5, -0.0001), (6, 0.0001), (6, -0.0001)
                                             ]:
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

            refine_list = [name]
            for itm in best_m:
                refine_list.append(itm)
            # 打开文件以追加方式写入
            with open('data/7scenes/stairs/results_refine.txt', 'a') as f:
                # 将列表中的每个元素转换为字符串并写入文件
                f.write(' '.join(map(str, refine_list)) + '\n')

    end = time.time()

    print('time:', end-start)

