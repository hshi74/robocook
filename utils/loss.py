import numpy as np
import open3d as o3d
import scipy
import scipy.optimize
import torch

from perception.pcd_utils import upsample
from utils.visualize import visualize_o3d


def chamfer(x, y, pkg="numpy"):
    if pkg == "numpy":
        # numpy implementation
        x = np.repeat(np.expand_dims(x, axis=1), y.shape[0], axis=1)  # x: [N, M, D]
        y = np.repeat(np.expand_dims(y, axis=0), x.shape[0], axis=0)  # y: [N, M, D]
        dis = np.linalg.norm(x - y, 2, axis=2)
        dis_xy = np.mean(np.min(dis, axis=1))  # dis_xy: mean over N
        dis_yx = np.mean(np.min(dis, axis=0))  # dis_yx: mean over M
    else:
        # torch implementation
        x = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
        y = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=2)  # dis: [N, M]
        dis_xy = torch.mean(torch.min(dis, dim=1)[0])  # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=0)[0])  # dis_yx: mean over M

    return dis_xy + dis_yx


def emd(x, y, pkg="numpy"):
    if pkg == "numpy":
        # numpy implementation
        x_ = np.repeat(np.expand_dims(x, axis=1), y.shape[0], axis=1)  # x: [N, M, D]
        y_ = np.repeat(np.expand_dims(y, axis=0), x.shape[0], axis=0)  # y: [N, M, D]
        cost_matrix = np.linalg.norm(x_ - y_, 2, axis=2)
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(
                cost_matrix, maximize=False
            )
        except:
            # pdb.set_trace()
            print("Error in linear sum assignment!")

        emd = np.mean(np.linalg.norm(x[ind1] - y[ind2], 2, axis=1))
    else:
        # torch implementation
        x_ = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
        y_ = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
        dis = torch.norm(torch.add(x_, -y_), 2, dim=2)  # dis: [N, M]
        cost_matrix = dis.detach().cpu().numpy()
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(
                cost_matrix, maximize=False
            )
        except:
            # pdb.set_trace()
            print("Error in linear sum assignment!")

        emd = torch.mean(torch.norm(torch.add(x[ind1], -y[ind2]), 2, dim=1))

    return emd


def hausdorff(x, y, pkg="numpy"):
    if pkg == "numpy":
        x = np.repeat(np.expand_dims(x, axis=1), y.shape[0], axis=1)  # x: [N, M, D]
        y = np.repeat(np.expand_dims(y, axis=0), x.shape[0], axis=0)  # y: [N, M, D]
        dis = np.linalg.norm(x - y, 2, axis=2)
        dis_xy = np.max(np.min(dis, axis=1))  # dis_xy: mean over N
        dis_yx = np.max(np.min(dis, axis=0))  # dis_yx: mean over M
    else:
        # torch implementation
        x = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
        y = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=2)  # dis: [N, M]
        dis_xy = torch.max(torch.min(dis, dim=1)[0])  # dis_xy: mean over N
        dis_yx = torch.max(torch.min(dis, dim=0)[0])  # dis_yx: mean over M

    return dis_xy + dis_yx


def p2g(x, size, soft, p_mass=1.0):
    batch = x.shape[0]
    grid_m = torch.zeros(batch, size * size * size, device=x.device)
    # base = (x * size - 0.5).long()
    # fx = x * size - base.float()
    inv_dx = size
    fx = x * inv_dx
    base = (x * inv_dx - 0.5).long()
    fx = fx - base.float()

    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w[i][..., 0] * w[j][..., 1] * w[k][..., 2] * p_mass
                target = (base + torch.LongTensor([i, j, k], device=x.device)).clamp(
                    0, size - 1
                )
                idx = (target[..., 0] * size + target[..., 1]) * size + target[..., 2]
                grid_m.scatter_add_(1, idx, weight)

    if not soft:
        grid_m = (grid_m > 0.0001).float()

    return grid_m.reshape(batch, size, size, size)


def soft_iou(
    x, y, size=16, pkg="numpy", dense=True, soft=False, normalize=False, visualize=False
):
    if pkg == "numpy":
        x, y = torch.FloatTensor(x), torch.FloatTensor(y)

    if x.dim() == 2:
        x = x[None, :]

    if y.dim() == 2:
        y = y[None, :]

    if not dense:
        x_dense = []
        y_dense = []
        for i in range(x.shape[0]):
            x_dense.append(upsample(x[i], alpha=0.01, visualize=False))
            y_dense.append(upsample(y[i], alpha=0.01, visualize=False))
        x = torch.FloatTensor(x_dense)
        y = torch.FloatTensor(y_dense)

    if normalize:
        x = (x - torch.mean(x, dim=1)) / torch.std(x, dim=1)
        y = (y - torch.mean(y, dim=1)) / torch.std(y, dim=1)

    if visualize:
        x_pcd = o3d.geometry.PointCloud()
        x_pcd.points = o3d.utility.Vector3dVector(x.squeeze().numpy())
        x_pcd.paint_uniform_color([0, 0, 0.8])
        y_pcd = o3d.geometry.PointCloud()
        y_pcd.points = o3d.utility.Vector3dVector(y.squeeze().numpy())
        x_vg = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            x_pcd, 1.0 / size, x_pcd.get_min_bound(), x_pcd.get_max_bound()
        )
        y_vg = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            y_pcd, 1.0 / size, y_pcd.get_min_bound(), y_pcd.get_max_bound()
        )
        visualize_o3d([x_vg, y_vg])

    grid_x = p2g(x, size, soft)
    grid_y = p2g(y, size, soft)

    if soft:
        grid_x /= x.shape[1]
        grid_y /= y.shape[1]

    intersection = (grid_x * grid_y).sum()
    union = (grid_x + grid_y - grid_x * grid_y).sum()
    siou = intersection / union

    # import pdb; pdb.set_trace()
    if pkg == "numpy":
        siou = siou.detach().cpu().numpy()

    return siou


def iou(x, y, voxel_size=0.01, pkg="numpy", visualize=False):
    # x: [N, D]
    # y: [N, D]

    if visualize:
        x_pcd = o3d.geometry.PointCloud()
        x_pcd.points = o3d.utility.Vector3dVector(x)
        x_pcd.paint_uniform_color([0, 0, 0.8])
        y_pcd = o3d.geometry.PointCloud()
        y_pcd.points = o3d.utility.Vector3dVector(y)
        x_vg = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            x_pcd, voxel_size, x_pcd.get_min_bound(), x_pcd.get_max_bound()
        )
        y_vg = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            y_pcd, voxel_size, y_pcd.get_min_bound(), y_pcd.get_max_bound()
        )
        visualize_o3d([x_vg, y_vg])

    if pkg == "numpy":
        xy = np.concatenate([x, y], axis=0)
        # min_bound = np.floor(np.min(xy, axis=0))
        # max_bound = np.ceil(np.max(xy, axis=0))
        min_bound = np.floor(np.min(xy, axis=0) * 10) / 10
        max_bound = np.ceil(np.max(xy, axis=0) * 10) / 10

        grid_x = np.zeros(
            [
                int((max_bound[0] - min_bound[0]) / voxel_size) + 1,
                int((max_bound[1] - min_bound[1]) / voxel_size) + 1,
                int((max_bound[2] - min_bound[2]) / voxel_size) + 1,
            ]
        )

        grid_y = np.zeros_like(grid_x)
        # import pdb; pdb.set_trace()
        for i in range(x.shape[0]):
            x1 = int(np.floor((x[i][0] - min_bound[0]) / voxel_size))
            x2 = int(np.floor((x[i][1] - min_bound[1]) / voxel_size))
            x3 = int(np.floor((x[i][2] - min_bound[2]) / voxel_size))
            grid_x[x1, x2, x3] = 1
            # print('x1x2x3', x1, x2, x3)

        for i in range(y.shape[0]):
            y1 = int(np.floor((y[i][0] - min_bound[0]) / voxel_size))
            y2 = int(np.floor((y[i][1] - min_bound[1]) / voxel_size))
            y3 = int(np.floor((y[i][2] - min_bound[2]) / voxel_size))
            grid_y[y1, y2, y3] = 1
            # print('y1y2y3', y1, y2, y3)

        intersection = grid_x * grid_y
        union = grid_x + grid_y - grid_x * grid_y
        iou = np.sum(intersection) / np.sum(union)
    else:
        xy = torch.cat([x, y], dim=0)
        # min_bound = np.floor(np.min(xy, axis=0))
        # max_bound = np.ceil(np.max(xy, axis=0))
        min_bound = torch.floor(torch.min(xy, dim=0)[0] * 10) / 10
        max_bound = torch.ceil(torch.max(xy, dim=0)[0] * 10) / 10

        grid_x = torch.zeros(
            [
                int((max_bound[0] - min_bound[0]) / voxel_size) + 1,
                int((max_bound[1] - min_bound[1]) / voxel_size) + 1,
                int((max_bound[2] - min_bound[2]) / voxel_size) + 1,
            ]
        )

        grid_y = torch.zeros_like(grid_x)
        # import pdb; pdb.set_trace()
        for i in range(x.shape[0]):
            x1 = int(torch.floor((x[i][0] - min_bound[0]) / voxel_size))
            x2 = int(torch.floor((x[i][1] - min_bound[1]) / voxel_size))
            x3 = int(torch.floor((x[i][2] - min_bound[2]) / voxel_size))
            grid_x[x1, x2, x3] = 1
            # print('x1x2x3', x1, x2, x3)

        for i in range(y.shape[0]):
            y1 = int(torch.floor((y[i][0] - min_bound[0]) / voxel_size))
            y2 = int(torch.floor((y[i][1] - min_bound[1]) / voxel_size))
            y3 = int(torch.floor((y[i][2] - min_bound[2]) / voxel_size))
            grid_y[y1, y2, y3] = 1
            # print('y1y2y3', y1, y2, y3)

        intersection = grid_x * grid_y
        union = grid_x + grid_y - grid_x * grid_y
        iou = torch.sum(intersection) / torch.sum(union)

    return iou


if __name__ == "__main__":
    x = np.random.randn(500, 3)
    y = np.random.randn(500, 3)
    y1 = x + np.random.randn(500, 3) * 0.001

    # chamfer_loss = chamfer(x, y)
    # emd_loss = emd(x, y)
    # iou_loss = iou(x, y1, voxel_size=0.05)
    # siou_loss = soft_iou(x, y, size=64, soft=True)
    # siou_loss2 = soft_iou(x, y1, size=64, soft=True)
    # siou_loss3 = soft_iou(x, y, size=64, soft=False)
    # siou_loss4 = soft_iou(x, y1, size=64, soft=False)
    # siou_loss5 = soft_iou(x, y, size=128, soft=True)
    # siou_loss6 = soft_iou(x, y1, size=128, soft=True)
    # print(chamfer_loss, emd_loss, iou_loss, siou_loss, siou_loss2, siou_loss3, siou_loss4)
    # print('size', siou_loss5, siou_loss6)

    x_tensor = torch.FloatTensor(x).cuda()
    y_tensor = torch.FloatTensor(y).cuda()
    y1_tensor = torch.FloatTensor(y1).cuda()

    # chamfer_loss = chamfer(x_tensor, y_tensor, pkg='torch')
    # emd_loss = emd(x_tensor, y_tensor, pkg='torch')
    # iou_loss = iou(x_tensor, y_tensor, pkg='torch')
    # siou_loss = soft_iou(x_tensor, y_tensor, size=64, pkg='torch', soft=False)
    # print(chamfer_loss, emd_loss, iou_loss, siou_loss)
