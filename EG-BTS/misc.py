'''
some tools simplify the code
yzx
'''
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
import torch.utils.data.distributed
from PIL import Image
from torchvision.transforms import ToTensor
import torch.nn.functional as torch_nn_func
import cv2
import math
import torchvision.transforms as transforms
from skimage.transform import resize as skresize

def normalize_to_01(tensor):
    # 找到张量中的最小值和最大值
    min_val = tensor.min()
    max_val = tensor.max()

    # 将张量中的值归一化到 [0, 1] 区间
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    return normalized_tensor
def save_one(path,image):
    if isinstance(image,torch.Tensor):
        image = image.detach().cpu().numpy().squeeze()

    plt.imsave(path,image,cmap='magma_r')


def dilate_edges_test(edges, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel)
    return dilated_edges

def dilate_edges(edge_tensor, kernel_size=5, iterations=1):
    # Define a kernel for dilation
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=edge_tensor.device)

    # Perform dilation
    dilated_edges = torch_nn_func.conv2d(edge_tensor, kernel, padding=kernel_size // 2, dilation=1, groups=1)

    # Apply dilation for multiple iterations
    for _ in range(iterations - 1):
        dilated_edges = torch_nn_func.conv2d(dilated_edges, kernel, padding=kernel_size // 2, dilation=1, groups=1)

    # Threshold the result to ensure binary values
    dilated_edges[dilated_edges > 0.2] = 1

    return dilated_edges
def apply_otsu_thresholding(image,output_path):
    # 将PyTorch张量转换为NumPy数组
    image_np = image.squeeze().detach().cpu().numpy()

    # 将图像转换为OpenCV格式
    image_cv = (image_np * 255).astype(np.uint8)

    # 使用OpenCV的OTSU阈值法
    _, thresholded = cv2.threshold(image_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 将二值化结果转换回PyTorch张量
    edge_mask = torch.from_numpy(thresholded / 255.0).unsqueeze(0).unsqueeze(0).float()

    return edge_mask


def vis_edge_mask(edge_mask1,edge_mask2,edge_mask):
    # 可视化掩码
    edge_mask_np1 = edge_mask1.squeeze().detach().cpu().numpy()
    edge_mask_np2 = edge_mask2.squeeze().detach().cpu().numpy()
    edge_mask_np = edge_mask.squeeze().detach().cpu().numpy()
    plt.figure(figsize=(16, 12))
    plt.subplot(1, 3, 1)
    plt.imshow(edge_mask_np1, cmap='gray')
    plt.axis('off')
    plt.title('0.5')

    plt.subplot(1, 3, 2)
    plt.imshow(edge_mask_np2, cmap='gray')
    plt.axis('off')
    plt.title('0.65')
    plt.subplot(1, 3, 3)
    plt.imshow(edge_mask_np, cmap='gray')
    plt.axis('off')
    plt.title('0.8')
    plt.savefig('edge_mask.png')
    plt.show()
def sobel_misc(image):
    is_tensor = True
    if not isinstance(image, torch.Tensor):
        # 如果数据类型不是PyTorch张量，则转换为张量
        image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        is_tensor = False
        # print("convert image shape",image.shape)
    device = image.device
    # 定义Sobel算子的卷积核
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32,device=image.device).view(1, 1, 3, 3)
    # 在图像上应用Sobel算子进行卷积操作
    gradient_x = torch_nn_func.conv2d(image, sobel_x, padding=1)
    gradient_y = torch_nn_func.conv2d(image, sobel_y, padding=1)

    # 计算梯度的幅度和角度
    gradient_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
    # gradient_angle = torch.atan2(gradient_y, gradient_x)
    # 如果输入不是张量，则将结果转换为NumPy数组
    if not is_tensor:
        gradient_magnitude = gradient_magnitude.squeeze().detach().numpy()

    return gradient_magnitude

def sobel_misc2(image):
    is_tensor = True
    if not isinstance(image, torch.Tensor):
        # 如果数据类型不是PyTorch张量，则转换为张量
        image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        is_tensor = False
        # print("convert image shape",image.shape)
    device = image.device
    # 定义Sobel算子的卷积核
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32,device=image.device).view(1, 1, 3, 3)
    # 在图像上应用Sobel算子进行卷积操作
    gradient_x = torch_nn_func.conv2d(image, sobel_x, padding=1)
    gradient_y = torch_nn_func.conv2d(image, sobel_y, padding=1)

    return gradient_x,gradient_y

def LoG_edge_detection(input_tensor, sigma=1.0):
    # Define LoG kernel
    kernel_size = 5
    device = input_tensor.device
    dtype = input_tensor.dtype
    x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    y = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    xx, yy = torch.meshgrid(x, y)
    kernel = (-1 / (math.pi * sigma ** 4)) * (1 - (xx ** 2 + yy ** 2) / (2 * sigma ** 2)) * torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel - torch.mean(kernel)
    kernel = kernel / torch.sum(torch.abs(kernel))
    kernel = kernel.to(dtype).to(device)

    # Apply LoG filter
    edge_map = torch_nn_func.conv2d(input_tensor, kernel.view(1, 1, kernel_size, kernel_size), padding=kernel_size // 2)

    return edge_map


def canny_edge_detection(input_tensor, low_threshold=50, high_threshold=150):
    batch_size, channels, height, width = input_tensor.size()

    edge_tensors = []
    for i in range(batch_size):
        # Convert tensor to numpy array
        input_array = input_tensor[i].cpu().numpy().transpose((1, 2, 0))

        # Convert to uint8 and scale to [0, 255]
        input_array = (input_array - input_array.min()) / (input_array.max() - input_array.min()) * 255
        input_array = input_array.astype('uint8')

        # Convert to grayscale
        # gray_image = cv2.cvtColor(input_array, cv2.COLOR_RGB2GRAY)

        # Perform Canny edge detection
        edges = cv2.Canny(input_array, low_threshold, high_threshold)

        # Convert numpy array back to tensor
        edge_tensor = torch.tensor(edges, dtype=input_tensor.dtype, device=input_tensor.device).unsqueeze(0).unsqueeze(0)
        edge_tensors.append(edge_tensor)

    return torch.cat(edge_tensors, dim=0)
def vis_edge2(sgt,simg,cgt,cimg,a,b,c,d, title_,path):
    # 可视化输入深度图
    plt.figure(figsize=(12, 9))
    # 第一行
    plt.subplot(3, 4, 1)
    plt.imshow(sgt.squeeze().cpu().numpy(), cmap='gray')
    plt.title(title_[0][0])
    plt.axis('off')
    # 可视化 Sobel 边缘检测结果
    plt.subplot(3, 4, 2)
    plt.imshow(simg.squeeze().cpu().numpy(), cmap='gray')
    plt.title(title_[0][1])
    plt.axis('off')
    # 可视化深度边缘掩码
    plt.subplot(3,4, 3)
    plt.imshow(cgt.squeeze().cpu().numpy(), cmap='gray')
    plt.title(title_[0][2])
    plt.axis('off')
    plt.subplot(3, 4, 4)
    plt.imshow(cimg.squeeze().cpu().numpy(), cmap='gray')
    plt.title(title_[0][3])
    plt.axis('off')
    # 第二行
    plt.subplot(3, 4, 5)
    plt.imshow(a.squeeze().cpu().numpy(), cmap='gray')
    plt.title(title_[1][0])
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.imshow(b.squeeze().cpu().numpy(), cmap='gray')
    plt.title(title_[1][1])
    plt.axis('off')

    # 可视化深度边缘掩码
    plt.subplot(3,4, 9)
    plt.imshow(c.squeeze().cpu().numpy(), cmap='gray')
    plt.title(title_[2][0])
    plt.axis('off')

    plt.subplot(3,4,10)
    plt.imshow(d.squeeze().cpu().numpy(), cmap='gray')
    plt.title(title_[2][1])
    plt.axis('off')
    plt.tight_layout()

    # 保存图像
    plt.savefig(path,dpi=100)

    # plt.show()

def vis_edge1(sgt,simg,cgt,cimg,path):
    # 可视化输入深度图
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.imshow(sgt.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Sobel gt')
    plt.axis('off')

    # 可视化 Sobel 边缘检测结果
    plt.subplot(1, 4, 2)
    plt.imshow(simg.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Sobel image')
    plt.axis('off')

    # 可视化深度边缘掩码
    plt.subplot(1,4, 3)
    plt.imshow(cgt.squeeze().cpu().numpy(), cmap='gray')
    plt.title('canny gt')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(cimg.squeeze().cpu().numpy(), cmap='gray')
    plt.title('canny image')
    plt.axis('off')
    plt.tight_layout()

    # 保存图像
    plt.savefig(path,dpi=100)

    # plt.show()

def vis_edge(input_depth,sobel_edge_map,depth_edge_mask):
    # 可视化输入深度图
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(input_depth.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Input Depth Map')
    plt.axis('off')

    # 可视化 Sobel 边缘检测结果
    plt.subplot(1, 3, 2)
    plt.imshow(sobel_edge_map.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')

    # 可视化深度边缘掩码
    plt.subplot(1, 3, 3)
    plt.imshow(depth_edge_mask.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Depth Edge Mask')
    plt.axis('off')

    plt.tight_layout()

    # 保存图像
    plt.savefig('visualization.png')

    plt.show()


def unnormalize_tensor(tensor):
    """
    将归一化后的张量反归一化到0到1的范围。

    参数：
        tensor (torch.Tensor): 归一化后的张量。
        mean (list): 均值。
        std (list): 标准差。

    返回：
        torch.Tensor: 反归一化后的张量。
    """
    mean_ = [0.485, 0.456, 0.406]
    std_ = [0.229, 0.224, 0.225]
    # 创建逆操作
    unnormalized = tensor.clone()
    for channel in range(3):  # 假设张量是RGB图像
        unnormalized[:, channel, :, :] = tensor[:, channel, :, :] * std_[channel] + mean_[channel]

    unnormalized[torch.isnan(unnormalized)] = 0

    # 限制在0到1的范围内
    unnormalized.clamp_(0, 1)

    # 反归一化
    # tensor_unnormalized = unnormalize(tensor)

    return unnormalized


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


def center_crop(image, size):
    h1, w1 = image.shape

    tw, th = size

    if w1 == tw and h1 == th:
        return image

    x1 = int(round((w1 - tw) / 2.))
    y1 = int(round((h1 - th) / 2.))

    image = image[y1:th + y1, x1:tw + x1]

    return image


def compute_edges_ok(gt, pred,mask):
    thre = 0.25
    gt[mask==False]=0
    edge_gt = sobel_misc(gt)
    # cv2.imwrite("vis_result/edge_gt_orin.jpg", edge_gt * 255)

    # edge_gt[mask == False] = 0
    # cv2.imwrite("vis_result/edge_gt_mask.jpg", edge_gt * 255)

    # plt.imsave("vis_result/edge_gt.jpg",edge_gt,cmap='gray')
    pred[mask == False] =0
    # plt.imsave("vis_result/pred_mask.jpg",pred,cmap='gray')
    edge_pred = sobel_misc(pred)
    # cv2.imwrite("vis_result/edge_edge_pred.jpg", edge_pred * 255)

    # 加入mask,

    # 直接比较
    edge_gt_valid = (edge_gt > thre)
    edge_pred_valid = (edge_pred > thre)



    nvalid = np.sum(np.equal(edge_gt_valid, edge_pred_valid))
    A = nvalid / (edge_gt.shape[0] * edge_gt.shape[1])

    nvalid2 = np.sum(((edge_gt_valid & edge_pred_valid) == True))
    P = nvalid2 / (np.sum(edge_pred_valid))
    R = nvalid2 / (np.sum(edge_gt_valid))

    if P + R == 0:
        F = 0
    else:
        F = (2 * P * R) / (P + R)

    return P, R, F

def downsample_image(input_image):
    # 假设输入图像是 NumPy 数组，形状为 (height, width, channels)
    height, width = input_image.shape[:2]
    # 目标尺寸是原来的四分之一
    target_height, target_width = height // 4, width // 4
    # 使用 resize 函数进行插值下采样
    downsampled_image = skresize(input_image, (target_height, target_width), mode='reflect', anti_aliasing=True)
    return downsampled_image

def compute_edges(gt, pred,mask,mode_e=None):
    thre = 0.25

    # 直接进行评估
    if mode_e == '1' or mode_e == None:
        # gt[mask==False]=0
        edge_gt = sobel_misc(gt)
        # cv2.imwrite("vis_result/edge_gt_orin.jpg", edge_gt * 255)
        # edge_gt[mask == False] = 0
        # cv2.imwrite("vis_result/edge_gt_mask.jpg", edge_gt * 255)

        # plt.imsave("vis_result/edge_gt.jpg",edge_gt,cmap='gray')
        # pred[mask == False] =0
        # plt.imsave("vis_result/pred_mask.jpg",pred,cmap='gray')
        edge_pred = sobel_misc(pred)

        # cv2.imwrite("vis_result/edge_edge_pred.jpg", edge_pred * 255)

        # 加入膨胀后比较,变成了0、1
        edge_gt_valid = (edge_gt > thre).astype(np.float32)
        edge_pred_valid = (edge_pred > thre).astype(np.float32)
        #
        edge_gt_valid = dilate_edges_test(edge_gt_valid).astype(bool)
        edge_pred_valid = dilate_edges_test(edge_pred_valid).astype(bool)
        # 直接比较
        # edge_gt_valid = (edge_gt > thre)
        # edge_pred_valid = (edge_pred > thre)
        # edge_gt_valid[mask==False]=0
        # edge_pred_valid[mask==False]=0

        nvalid = np.sum(np.equal(edge_gt_valid, edge_pred_valid))
        aa = nvalid / (edge_gt.shape[0] * edge_gt.shape[1])

        nvalid2 = np.sum(((edge_gt_valid & edge_pred_valid) == True))
        pp = nvalid2 / (np.sum(edge_pred_valid))
        rr = nvalid2 / (np.sum(edge_gt_valid))

        if pp + rr == 0:
            ff = 0
        else:
            ff = (2 * pp * rr) / (pp + rr)

    elif mode_e == '2':  # reshape成228，304的图形进行比较
        gt[mask==False]=0
        pred[mask == False] =0

        gt = center_crop(gt, (304, 228))
        pred = center_crop(pred, (304, 228))
        edge_gt = sobel_misc(gt)

        edge_pred = sobel_misc(pred)
        # cv2.imwrite("vis_result/edge_gt_orin.jpg", edge_gt * 255)
        # edge_gt[mask == False] = 0
        # cv2.imwrite("vis_result/edge_gt_mask.jpg", edge_gt * 255)

        # plt.imsave("vis_result/edge_gt.jpg",edge_gt,cmap='gray')
        # pred[mask == False] =0
        # plt.imsave("vis_result/pred_mask.jpg",pred,cmap='gray')
        # cv2.imwrite("vis_result/edge_edge_pred.jpg", edge_pred * 255)

        # # 加入膨胀后比较,变成了0、1
        # edge_gt_valid = (edge_gt > thre).astype(np.float32)
        # edge_pred_valid = (edge_pred > thre).astype(np.float32)
        # #
        # edge_gt_valid = dilate_edges_test(edge_gt_valid).astype(bool)
        # edge_pred_valid = dilate_edges_test(edge_pred_valid).astype(bool)
        # 直接比较
        edge_gt_valid = (edge_gt > thre)
        edge_pred_valid = (edge_pred > thre)

        nvalid = np.sum(np.equal(edge_gt_valid, edge_pred_valid))
        aa = nvalid / (edge_gt.shape[0] * edge_gt.shape[1])

        nvalid2 = np.sum(((edge_gt_valid & edge_pred_valid) == True))
        pp = nvalid2 / (np.sum(edge_pred_valid))
        rr = nvalid2 / (np.sum(edge_gt_valid))

        if pp + rr == 0:
            ff = 0
        else:
            ff = (2 * pp * rr) / (pp + rr)

    elif mode_e == '3':
        gt[mask == False] = 0
        edge_gt = sobel_misc(gt)
        save_gt = (edge_gt * 255).astype(np.uint8)
        cv2.imwrite("vis_result/edge_gt_orin.jpg", save_gt)
        # cv2.imwrite("vis_result/edge_gt_mask.jpg", edge_gt * 255)

        # plt.imsave("vis_result/edge_gt.jpg",edge_gt,cmap='gray')
        # pred[mask == False] =0
        # plt.imsave("vis_result/pred_mask.jpg",pred,cmap='gray')
        edge_pred = sobel_misc(pred)
        save_pre = (edge_pred * 255).astype(np.uint8)
        cv2.imwrite("vis_result/edge_edge_pred.jpg", save_pre)
        edge_gt_nor = (edge_gt > 0.25).astype(np.float32)
        edge_pred_nor = (edge_pred > 0.25).astype(np.float32)
        cv2.imwrite("vis_result/edge_gt_nor.jpg", edge_gt_nor*255)
        cv2.imwrite("vis_result/edge_pred_nor.jpg", edge_pred_nor*255)

        scale_ = round(edge_pred.max()/edge_gt.max(), 3)
        # 加入膨胀后比较,变成了0、1
        # edge_gt_valid = (edge_gt > thre).astype(np.float32)
        # edge_pred_valid = (edge_pred > thre).astype(np.float32)
        #
        # edge_gt_valid = dilate_edges_test(edge_gt_valid).astype(bool)
        # edge_pred_valid = dilate_edges_test(edge_pred_valid).astype(bool)
        # 直接比较
        edge_gt_valid = (edge_gt > thre)
        edge_pred_valid = (edge_pred > thre)

        nvalid = np.sum(np.equal(edge_gt_valid, edge_pred_valid))
        aa = nvalid / (edge_gt.shape[0] * edge_gt.shape[1])

        nvalid2 = np.sum(((edge_gt_valid & edge_pred_valid) == True))
        pp = nvalid2 / (np.sum(edge_pred_valid))
        rr = nvalid2 / (np.sum(edge_gt_valid))

        if pp + rr == 0:
            ff = 0
        else:
            ff = (2 * pp * rr) / (pp + rr)

    elif mode_e == '4':
        gt = downsample_image(gt)
        pred = downsample_image(pred)
        edge_gt = sobel_misc(gt)
        edge_pred = sobel_misc(pred)

        edge_gt_valid = (edge_gt > thre).astype(np.float32)
        edge_pred_valid = (edge_pred > thre).astype(np.float32)
        #
        edge_gt_valid = dilate_edges_test(edge_gt_valid).astype(bool)
        edge_pred_valid = dilate_edges_test(edge_pred_valid).astype(bool)
        # 直接比较
        # edge_gt_valid = (edge_gt > thre)
        # edge_pred_valid = (edge_pred > thre)
        # edge_gt_valid[mask==False]=0
        # edge_pred_valid[mask==False]=0

        nvalid = np.sum(np.equal(edge_gt_valid, edge_pred_valid))
        aa = nvalid / (edge_gt.shape[0] * edge_gt.shape[1])

        nvalid2 = np.sum(((edge_gt_valid & edge_pred_valid) == True))
        pp = nvalid2 / (np.sum(edge_pred_valid))
        rr = nvalid2 / (np.sum(edge_gt_valid))

        if pp + rr == 0:
            ff = 0
        else:
            ff = (2 * pp * rr) / (pp + rr)
    return pp, rr, ff

def color_cv(value, vmin=0, vmax=1, cmap='magma_r'):
    """Converts a depth map to a color image.
    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 3)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()

    # normalize
    vmin = np.percentile(value,2) if vmin is None else vmin
    vmax = np.percentile(value,85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

        # value = value[:,:,:]
        # plt.imsave("out.png", value)
    return (value*255).astype(np.uint8)


def color(path, value, vmin=0, vmax=10, cmap='magma_r'):
    """Converts a depth map to a color image.
    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 3)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()

    # normalize
    vmin = np.percentile(value,2) if vmin is None else vmin
    vmax = np.percentile(value,85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    if path is None:
        cmapper = matplotlib.cm.get_cmap(cmap)
        value = cmapper(value, bytes=True)  # (nxmx4)
        value = value[:,:,:]
        plt.imsave("out_uint.png", value)
        return value
    plt.imsave(path, value,cmap=cmap)
    return

def colors(values, vmin=None, vmax=None, cmap='magma_r',path=None):
    """Converts a depth map to a color image.
    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if len(values) > 1 and len(values) == len(path):
        for i in range(0,len(values)):
            if isinstance(values[i], torch.Tensor):
                value = values[i].detach().cpu().numpy()

            value = values[i].squeeze()

            # normalize
            vmin = np.percentile(value,2) if vmin is None else vmin
            vmax = np.percentile(value,85) if vmax is None else vmax
            if vmin != vmax:
                value = (value - vmin) / (vmax - vmin)  # vmin..vmax
            else:
                # Avoid 0-division
                value = value * 0.
            # print(path[i])
            plt.imsave(path[i], value,cmap=cmap)

def colorize(value, vmin=None, vmax=None, cmap='magma_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.data.cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img