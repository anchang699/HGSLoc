import os
import numpy as np
from PIL import Image
from typing import NamedTuple
from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel, BasicPointCloud
from arguments import ModelParams
from utils.graphics_utils import getWorld2View2, focal2fov
from scene.colmap_loader import qvec2rotmat, read_points3D_binary, read_points3D_text
from plyfile import PlyData, PlyElement
from utils.general_utils import PILtoTorch
from scene.cameras_shader import Camera

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height

    image = Image.new('RGB', (orig_w, orig_h), color='white')

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(image, resolution)

    size_image = resized_image_rgb[:3, ...]

    return Camera(R=cam_info.R, T=cam_info.T, FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  width=size_image.shape[2], height=size_image.shape[1], data_device=args.data_device)


class CameraInfo(NamedTuple):
    # uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    # image: np.array
    # image_path: str
    # image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    ply_path: str

# def getNerfppNorm(cam_info):
#     def get_center_and_diag(cam_centers):
#         cam_centers = np.hstack(cam_centers)
#         avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
#         center = avg_cam_center
#         dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
#         diagonal = np.max(dist)
#         return center.flatten(), diagonal
#
#     cam_centers = []
#
#     for cam in cam_info:
#         W2C = getWorld2View2(cam.R, cam.T)
#         C2W = np.linalg.inv(W2C)
#         cam_centers.append(C2W[:3, 3:4])
#
#     center, diagonal = get_center_and_diag(cam_centers)
#     radius = diagonal * 1.1
#
#     translate = -center
#
#     return {"translate": translate, "radius": radius}


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def readColmapSceneInfo(path, height, width, qandt,
                 cam_intrinsic_first, cam_intrinsic_second):

    height = height
    width = width

    # SIMPLE_PINHOLE/SIMPLE_RADIAL
    focal_length_x = cam_intrinsic_first
    FovY = focal2fov(focal_length_x, height)
    FovX = focal2fov(focal_length_x, width)

    ## PINHOLE
    # focal_length_x = cam_intrinsic_first
    # focal_length_y = cam_intrinsic_second
    # FovY = focal2fov(focal_length_y, height)
    # FovX = focal2fov(focal_length_x, width)

    # ### 随便赋值
    # # 在训练库图像中随便选一张
    # image_path = "D:\PyCharmProject\comment_3DGS\data\chess\images\\seq-01-frame-000004.color.png"
    # image_name = "DSC05612"
    # # 创建一个与输入图像相同大小的背景图像
    # image = Image.new('RGB', (width, height), color='white')
    # ###

    cam_infos = []
    for pose_item in qandt:
        qvec = np.array(pose_item[:4])
        tvec = np.array(pose_item[4:])

        R = np.transpose(qvec2rotmat(qvec))
        T = np.array(tvec)

        cam_info = CameraInfo(R=R, T=T, FovY=FovY, FovX=FovX, width=width, height=height)
        cam_infos.append(cam_info)

    train_cam_info = cam_infos
    test_cam_info = []


    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_info,
                           test_cameras=test_cam_info,
                           ply_path=ply_path)
    return scene_info

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

class SceneShader:

    # 初始化方法
    def __init__(self, args: ModelParams, height, width, qandt,
                 cam_intrinsic_first, cam_intrinsic_second, load_iteration=None, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """

        self.train_cameras = {}

        # 根据场景的类型（Colmap 或 Blender）加载相应的场景信息，存储在 scene_info 变量中。
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            scene_info = readColmapSceneInfo(args.source_path, height, width, qandt,
                 cam_intrinsic_first, cam_intrinsic_second)


        # 加载训练和测试相机：
        for resolution_scale in resolution_scales:  # 可选参数，默认为 [1.0]。一个浮点数列表，用于指定训练和测试相机的分辨率缩放因子。
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)


    def getTrainCameras(self, scale=1.0):  # 返回训练相机的列表，可以根据指定的缩放因子 scale 获取相应分辨率的相机列表。
        # pop() 方法从列表中删除并返回指定索引（默认是最后一个元素）的元素。
        return self.train_cameras[scale]  # train_cameras是字典结构，字典的键值是列表结构，可以执行pop操作
