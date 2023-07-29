import copy
import numpy as np
import open3d as o3d
import pymeshfix
import pyvista as pv
import scipy
import torch

from collections import defaultdict
from itertools import product
from pysdf import SDF
from pytorch3d.transforms import quaternion_to_matrix
from transforms3d.quaternions import *
from utils.visualize import *


def length(x_arr):
    return np.array([np.sqrt(x.dot(x) + 1e-8) for x in x_arr])


def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


def alpha_shape_mesh_reconstruct(pcd, alpha=0.5, mesh_fix=False, visualize=False):
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha, tetra_mesh, pt_map
    )

    if mesh_fix:
        mf = pymeshfix.MeshFix(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
        mf.repair()
        mesh = mf.mesh

    if visualize:
        if mesh_fix:
            plt = pv.Plotter()
            point_cloud = pv.PolyData(np.asarray(pcd.points))
            plt.add_mesh(point_cloud, color="k", point_size=10)
            plt.add_mesh(mesh)
            plt.add_title("alpha_shape_reconstruction")
            plt.show()
        else:
            visualize_o3d([pcd, mesh], title="alpha_shape_reconstruction")

    return mesh


def ball_pivoting_mesh_reconstruct(
    pcd, radii=[0.001, 0.002, 0.004, 0.008], visualize=False
):
    pcd = segment_and_filp(pcd, visualize=visualize)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )

    mf = pymeshfix.MeshFix(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    mf.repair()
    pymesh = mf.mesh

    if visualize:
        plt = pv.Plotter()
        point_cloud = pv.PolyData(np.asarray(pcd.points))
        plt.add_mesh(point_cloud, color="k", point_size=10)
        plt.add_mesh(pymesh)
        plt.add_title("alpha_shape_reconstruction")
        plt.show()
        # visualize_o3d([pcd, mesh], title='alpha_shape_reconstruction')

    return pymesh


def vista_mesh_reconstruct(pcd, visualize=False):
    point_cloud = pv.PolyData(np.asarray(pcd.points))
    surf = point_cloud.reconstruct_surface()

    mf = pymeshfix.MeshFix(surf)
    mf.repair()
    pymesh = mf.mesh

    if visualize:
        plt = pv.Plotter()
        plt.add_mesh(point_cloud, color="k", point_size=10)
        plt.add_mesh(pymesh)
        plt.add_title("Reconstructed Surface")
        plt.show()

    return pymesh


def flip_all_inward_normals(pcd, center, threshold=0.7):
    # Flip normal if normal points inwards by changing vertex order
    # https://math.stackexchange.com/questions/3114932/determine-direction-of-normal-vector-of-convex-polyhedron-in-3d

    # Get vertices and triangles from the mesh
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    # For each triangle in the mesh
    # flipped_count = 0
    for i, n in enumerate(normals):
        # Compute vector from 1st vertex of triangle to center
        norm_ref = points[i] - center
        # Compare normal to the vector
        if np.dot(norm_ref, n) < 0:
            # Change vertex order to flip normal direction
            # flipped_count += 1
            # if flipped_count > threshold * normals.shape[0]:
            normals[i] = np.negative(normals[i])
            # break

    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def segment_and_filp(pcd, visualize=False):
    labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=100))
    part_one = pcd.select_by_index(np.where(labels == 0)[0])
    part_two = pcd.select_by_index(np.where(labels > 0)[0])

    for part in [part_one, part_two]:
        if np.asarray(part.points).shape[0] > 0:
            # invalidate existing normals
            part.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
            part.estimate_normals()
            part.orient_normals_consistent_tangent_plane(100)

            # get an accurate center
            # center = part.get_center()
            hull, _ = part.compute_convex_hull()
            center = hull.get_center()

            part = flip_all_inward_normals(part, center)

            if visualize:
                visualize_o3d([part, hull], title="part_normals", show_normal=True)

    return part_one + part_two


# @profile
def poisson_mesh_reconstruct(cube, rest=None, depth=8, mesh_fix=False, visualize=False):
    cube.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    cube.estimate_normals()
    cube.orient_normals_consistent_tangent_plane(100)

    # get an accurate center
    hull, _ = cube.compute_convex_hull()
    center = hull.get_center()

    cube = flip_all_inward_normals(cube, center)

    if rest is None:
        pcd = cube
    else:
        rest = segment_and_filp(rest, visualize=visualize)
        pcd = cube + rest

    # n_threads = 1 if this is during close loop control
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=1
    )

    if mesh_fix:
        mf = pymeshfix.MeshFix(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
        mf.repair()
        mesh = mf.mesh

    if visualize:
        if mesh_fix:
            plt = pv.Plotter()
            point_cloud = pv.PolyData(np.asarray(pcd.points))
            plt.add_mesh(point_cloud, color="k", point_size=10)
            plt.add_mesh(mesh)
            plt.add_title("poisson_reconstruction")
            plt.show()
        else:
            visualize_o3d([pcd, mesh], title="possion_reconstruction")

    return mesh


def vg_filter(cube, sampled_points, grid_size=0.001, axis=2, visualize=False):
    lower = cube.get_min_bound()
    upper = cube.get_max_bound()
    cube_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cube, 0.9 * grid_size)

    if visualize:
        visualize_o3d([cube, cube_grid], title="voxel_grid_filter")

    ax_grid = np.arange(lower[axis], upper[axis], grid_size)
    tiled_points = np.tile(sampled_points[:, None, :], (1, ax_grid.shape[0], 1))
    test_points = copy.deepcopy(tiled_points)
    test_points[:, :, axis] = ax_grid
    exists_mask = np.array(
        cube_grid.check_if_included(
            o3d.utility.Vector3dVector(test_points.reshape((-1, 3)))
        )
    )
    exists_mask = exists_mask.reshape((-1, ax_grid.shape[0]))
    if axis == 2:
        vg_up_mask = (
            np.sum((tiled_points[:, :, axis] < ax_grid) * exists_mask, axis=1) > 0
        )
        vg_down_mask = (
            np.sum((tiled_points[:, :, axis] > ax_grid) * exists_mask, axis=1) > 0
        )

        # if n_points_close > 0:
        #     return (vg_up_mask & ~vg_down_mask) | (~vg_up_mask & ~vg_down_mask & cube_mesh_mask)
        # else:
        return vg_up_mask & ~vg_down_mask
    else:
        raise NotImplementedError


def upsample(points, sample_size=20000, alpha=0.1, visualize=False):
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    lower = pcd.get_min_bound()
    upper = pcd.get_max_bound()
    sampled_points = np.random.rand(sample_size, 3) * (upper - lower) + lower

    mesh = alpha_shape_mesh_reconstruct(pcd, alpha=alpha, visualize=visualize)

    f = SDF(mesh.vertices, mesh.triangles)

    sdf = f(sampled_points)
    sampled_points = sampled_points[-sdf < 0, :]

    if visualize:
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        sampled_pcd.paint_uniform_color([0.0, 0.0, 1.0])
        # vg = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(sampled_pcd, 0.01, lower, upper)
        visualize_o3d([sampled_pcd])

    return sampled_points


def fps(pts, n_particles=300):
    farthest_pts = np.zeros((n_particles, 3))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, n_particles):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts


def pairwise_registration(source, target, max_correspondence_distance_fine):
    # print("Apply point-to-plane ICP")
    # icp_coarse = o3d.pipelines.registration.registration_icp(
    #     source, target, max_correspondence_distance_coarse, np.identity(4),
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_fine,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    transformation_icp = icp_fine.transformation
    information_icp = (
        o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance_fine, icp_fine.transformation
        )
    )
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_fine
            )
            # print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=False,
                    )
                )
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=True,
                    )
                )
    return pose_graph


def o3d_registration(
    pcds_down, max_correspondence_distance_fine=0.002, visualize=False
):
    print("Full registration...")

    for pcd in pcds_down:
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)

    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(pcds_down, max_correspondence_distance_fine)

    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds_down)):
        # if visualize:
        #     visualize_o3d([pcds_down[point_id]], title='partial_point_cloud')
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds_down[point_id]

    if visualize:
        visualize_o3d([pcd_combined], title="merged_point_cloud")

    return pcd_combined, pose_graph


def scalable_integrate_rgb_frames(args, rgbd_images, visualize=False):
    poses = []
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=1.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    pose_graph_rgbd = o3d.io.read_pose_graph("misc/pose_graph.json")

    depth_T = np.concatenate(
        (
            np.concatenate(
                (
                    quat2mat(args.depth_optical_frame_pose[3:]),
                    np.array([args.depth_optical_frame_pose[:3]]).T,
                ),
                axis=1,
            ),
            [[0, 0, 0, 1]],
        )
    )

    for i in range(len(pose_graph_rgbd.nodes)):
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(
            f"perception/config/intrinsics_{i+1}.json"
        )
        cam_T = np.concatenate(
            (
                np.concatenate(
                    (
                        quat2mat(args.cam_pose_dict[f"cam_{i+1}"]["orientation"]),
                        np.array([args.cam_pose_dict[f"cam_{i+1}"]["position"]]).T,
                    ),
                    axis=1,
                ),
                [[0, 0, 0, 1]],
            )
        )
        pose = cam_T @ depth_T @ pose_graph_rgbd.nodes[i].pose
        # import pdb; pdb.set_trace()
        volume.integrate(rgbd_images[i], intrinsic, np.linalg.inv(pose))

        # if visualize:
        #     visualize_o3d([rgbd_images[i]], title="rgbd_image")

        poses.append(pose)

    mesh = volume.extract_triangle_mesh()
    # voxel_pcd = volume.extract_voxel_point_cloud()
    # pcd = volume.extract_point_cloud()

    if visualize:
        visualize_o3d([mesh], title="merged_point_cloud")


########### UNUSED BELOW ##########
def alpha_shape_3D(pos, alpha):
    """
    Compute the alpha shape (concave hull) of a set of 3D points.
    Parameters:
        pos - np.array of shape [B, M, D] points.
        alpha - alpha value.
    return
        outer surface vertex indices, edge indices, and triangle indices
    """
    tetra = scipy.spatial.Delaunay(pos)
    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = np.take(pos, tetra.vertices, axis=0)
    normsq = np.sum(tetrapos**2, axis=2)[:, :, None]
    ones = np.ones((tetrapos.shape[0], tetrapos.shape[1], 1))
    a = np.linalg.det(np.concatenate((tetrapos, ones), axis=2))
    Dx = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [1, 2]], ones), axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 2]], ones), axis=2))
    Dz = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 1]], ones), axis=2))
    c = np.linalg.det(np.concatenate((normsq, tetrapos), axis=2))
    r = np.sqrt(Dx**2 + Dy**2 + Dz**2 - 4 * a * c) / (2 * np.abs(a))

    # Find tetrahedrals
    tetras = tetra.vertices[r < alpha, :]
    # triangles
    TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    Triangles = tetras[:, TriComb].reshape(-1, 3)
    Triangles = np.sort(Triangles, axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = defaultdict(int)
    for tri in Triangles:
        TrianglesDict[tuple(tri)] += 1
    Triangles = np.array([tri for tri in TrianglesDict if TrianglesDict[tri] == 1])
    # edges
    EdgeComb = np.array([(0, 1), (0, 2), (1, 2)])
    Edges = Triangles[:, EdgeComb].reshape(-1, 2)
    Edges = np.sort(Edges, axis=1)
    Edges = np.unique(Edges, axis=0)

    Vertices = np.unique(Edges)

    return Vertices


def compute_sdf(density, eps=1e-4, inf=1e10):
    if density.dim() == 3:
        density = density[None, :, :]
    dx = 1.0 / density.shape[1]
    with torch.no_grad():
        nearest_points = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(density.shape[1]),
                    torch.arange(density.shape[2]),
                    torch.arange(density.shape[3]),
                ),
                axis=-1,
            )[None, :]
            .to(density.device)
            .expand(density.shape[0], -1, -1, -1, -1)
            * dx
        )
        mesh_points = nearest_points.clone()

        is_object = (density <= eps) * inf
        sdf = is_object.clone()

        for i in range(density.shape[1] * 2):  # np.sqrt(1^2+1^2+1^2)
            for x, y, z in product(range(3), range(3), range(3)):
                if x + y + z == 0:
                    continue

                def get_slice(a):
                    if a == 0:
                        return slice(None), slice(None)
                    if a == 1:
                        return slice(0, -1), slice(1, None)
                    return slice(1, None), slice(0, -1)

                f1, t1 = get_slice(x)
                f2, t2 = get_slice(y)
                f3, t3 = get_slice(z)
                fr = (slice(None), f1, f2, f3)
                to = (slice(None), t1, t2, t3)
                dist = ((mesh_points[to] - nearest_points[fr]) ** 2).sum(axis=-1) ** 0.5
                dist += (sdf[fr] >= inf) * inf
                sdf_to = sdf[to]
                mask = (dist < sdf_to).float()
                sdf[to] = mask * dist + (1 - mask) * sdf_to
                mask = mask[..., None]
                nearest_points[to] = (1 - mask) * nearest_points[
                    to
                ] + mask * nearest_points[fr]
        return sdf


def p2v(xyz):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(xyz.shape[0]):
    pcd = o3d.geometry.PointCloud()
    # import pdb; pdb.set_trace()
    # print(xyz.shape)
    pcd.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.04)
    # o3d.visualization.draw_geometries([voxel_grid])
    # data = voxel_grid.create_dense(origin=[0,0,0], color=[0,0,0], voxel_size=0.03, width=1, height=1, depth=1)
    my_voxel = np.zeros((32, 32, 32))
    for j, d in enumerate(voxel_grid.get_voxels()):
        # print(j)
        my_voxel[d.grid_index[0], d.grid_index[1], d.grid_index[2]] = 1
        # z, x, y = my_voxel.nonzero()
        # ax.scatter(x, y, z, c=z, alpha=1)
        # plt.show()
    return torch.from_numpy(my_voxel).cuda()


# def pointcloud_align(pcd1, pcd2, debug_mat, debug_quat):
#     from utils.loss import chamfer, emd, soft_iou
#     loss1 = chamfer
#     loss2 = emd
#     loss3 = soft_iou
#     quaternion = torch.tensor(np.random.randn(4,)*0.001, requires_grad=True)
#     #torch.tensor(debug_quat+ np.random.randn(4,)*0.01, requires_grad=True) #torch.tensor(np.random.randn(4,), requires_grad=True)
#     if True:
#         torch.optim.Adam([quaternion], lr=.1, betas=(0.9, 0.999))
#     else:
#         optimizer = torch.optim.SGD([quaternion], lr=.01, momentum=0.8)
#     for i in range(1000):
#         optimizer.zero_grad()
#         unit_quat = quaternion / torch.norm(quaternion)
#         transform_matrix = quaternion_to_matrix(unit_quat).float()
#         pcd1_transform = pcd1 @ transform_matrix
#         pcd_loss = loss1(pcd1_transform, pcd2, pkg='torch')
#         pcd_loss.backward()
#         optimizer.step()
#         ### for validation
#         val_loss = loss1(pcd2, pcd2, pkg='torch')
#         val_loss_debug = loss1(pcd1@debug_mat, pcd2, pkg='torch')
#         print(pcd_loss.item(), unit_quat.detach())
#         print('val_loss', val_loss)
#         print('debug_loss', val_loss_debug)
#     return pcd_loss.item(), unit_quat, transform_matrix


if __name__ == "__main__":
    pcd1 = torch.randn(500, 3)
    random_quat = torch.tensor(
        np.random.randn(
            4,
        )
    )
    unit_quat = random_quat / torch.norm(random_quat)
    gt_mat = quaternion_to_matrix(unit_quat).float()
    pcd2 = pcd1 @ gt_mat
    # loss, output_quat, mat = pointcloud_align(pcd1, pcd2, gt_mat, random_quat)
    # print('gt quat', unit_quat.detach())
    # print(gt_mat.detach())
    # print(mat.detach())
    # import pdb; pdb.set_trace()
