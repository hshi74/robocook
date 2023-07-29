import copy
import cv2 as cv
import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np
import open3d as o3d
import os
import pickle
import pymeshfix
import pyvista as pv
import sys
import torch
import torchvision.transforms as transforms

from datetime import datetime
from sklearn import metrics
from trimesh import PointCloud

matplotlib.rcParams["legend.loc"] = "lower right"
color_list = ["royalblue", "red", "green", "cyan", "orange", "pink", "tomato", "violet"]


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def plot_train_loss(loss_dict, path=""):
    plt.figure(figsize=[16, 9])

    for label, loss in loss_dict.items():
        if not "loss" in label and not "accuracy" in label:
            continue
        time_list = list(range(len(loss)))
        plt.plot(time_list, loss, linewidth=6, label=label)

    plt.xlabel("epoches", fontsize=30)
    plt.ylabel("loss", fontsize=30)
    plt.title("Training Loss", fontsize=35)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def plot_eval_loss(
    title,
    loss_dict,
    loss_std_dict=None,
    alpha_fill=0.3,
    colors=None,
    path="",
    xlabel="Time / Steps",
    ylabel="Loss",
):
    plt.figure(figsize=[16, 9])

    if not colors:
        colors = color_list

    i = 0
    for label, loss in loss_dict.items():
        time_list = list(range(len(loss)))
        plt.plot(
            time_list, loss, linewidth=6, label=label, color=colors[i % len(colors)]
        )

        # plt.annotate(str(round(loss[0], 4)), xy=(0, loss[0]), xytext=(-30, 20), textcoords="offset points", fontsize=20)
        # plt.annotate(str(round(loss[-1], 4)), xy=(len(loss)-1, loss[-1]), xytext=(-30, 20), textcoords="offset points", fontsize=20)

        if loss_std_dict:
            loss_min_bound = loss - loss_std_dict[label]
            loss_max_bound = loss + loss_std_dict[label]
            plt.fill_between(
                time_list,
                loss_max_bound,
                loss_min_bound,
                color=colors[i % len(colors)],
                alpha=alpha_fill,
            )

        i += 1

    plt.xlabel(xlabel, fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    plt.title(title, fontsize=35)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def visualize_points(
    ax,
    args,
    particles,
    draw_set,
    target,
    mask=None,
    rels=None,
    axis_off=False,
    focus=True,
    res="high",
):
    ax.computed_zorder = False
    tool_dim = particles.shape[0] - args.n_particles - args.floor_dim
    point_size = 160 if res == "high" else 10
    outputs = []
    if "dough" in draw_set:
        if mask is None:
            c = "b"
        else:
            c = mask
        points = ax.scatter(
            particles[: args.n_particles, args.axes[0]],
            particles[: args.n_particles, args.axes[1]],
            particles[: args.n_particles, args.axes[2]],
            c=c,
            s=point_size,
        )
        outputs.append(points)
    if "floor" in draw_set:
        floor = ax.scatter(
            particles[
                args.n_particles : args.n_particles + args.floor_dim, args.axes[0]
            ],
            particles[
                args.n_particles : args.n_particles + args.floor_dim, args.axes[1]
            ],
            particles[
                args.n_particles : args.n_particles + args.floor_dim, args.axes[2]
            ],
            c="r",
            s=point_size,
        )
        outputs.append(floor)
    if "tool" in draw_set:
        tool = ax.scatter(
            particles[args.n_particles + args.floor_dim :, args.axes[0]],
            particles[args.n_particles + args.floor_dim :, args.axes[1]],
            particles[args.n_particles + args.floor_dim :, args.axes[2]],
            c="r",
            s=point_size,
            zorder=4.2,
        )
        outputs.append(tool)

    # if rels is not None and 'dough' in draw_set and 'tool' in draw_set and tool_dim > 0:
    #     if args.full_repr:
    #         dough_color = [(0., 0., 1., 1.)] * args.n_particles
    #         tool_color = [(1., 0., 0., 1.)] * tool_dim
    #         for i in range(rels.shape[0]):
    #             if rels[i][0] > 0 and rels[i][1] > 0:
    #                 for rel_idx in rels[i]:
    #                     if rel_idx < args.n_particles:
    #                         dough_color[rel_idx] = (0., 1., 0., 1.)
    #                     else:
    #                         tool_color[rel_idx - args.n_particles - args.floor_dim] = (0., 1., 0., 1.)
    #         outputs[0].set(edgecolor=dough_color)
    #         outputs[-1].set(edgecolor=tool_color)
    #     else:
    #         for i in range(rels.shape[0]):
    #             if rels[i][0] > 0 and rels[i][1] > 0:
    #                 neighbor = ax.plot(particles[rels[i], args.axes[0]], particles[rels[i], args.axes[1]],
    #                     particles[rels[i], args.axes[2]], c='g')
    #             else:
    #                 neighbor = ax.plot([], [], [], c='g')
    #             outputs.append(neighbor[0])

    # if args.full_repr and 'roller' in args.env and 'tool' in draw_set and tool_dim > 0:
    #     # roller_center = np.mean(particles[args.n_particles+args.floor_dim:], axis=0)
    #     # dist = np.linalg.norm(particles[args.n_particles+args.floor_dim:] - \
    #     #     np.tile(roller_center, (tool_dim, 1)), axis=0)
    #     # farthest_point_idx = np.argmax(dist)
    #     tool_color = [(1., 0., 0., 1.)] * tool_dim
    #     tool_color[0] = (0., 1., 1., 1.)
    #     tool_color[1] = (0., 1., 1., 1.)
    #     if 'large' in args.env or args.stage == 'dy':
    #         tool_color[2] = (0., 1., 1., 1.)
    #     else:
    #         tool_color[42] = (0., 1., 1., 1.)
    #     # for c in range(len(tool_color)):
    #     #     if c % 210 == 0:
    #     #         tool_color[c] = (0., 1., 1., 1.)
    #     outputs[-1].set(color=tool_color)

    if target is not None:
        ax.scatter(
            target[:, args.axes[0]],
            target[:, args.axes[1]],
            target[:, args.axes[2]],
            c="c",
            s=point_size,
            alpha=0.3,
        )

    if axis_off:
        ax.axis("off")

    # render high_res
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])

    # extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    # size = extents[:, 1] - extents[:, 0]
    centers = np.mean(np.array(particles[: args.n_particles]), axis=0)
    centers = [centers[args.axes[0]], centers[args.axes[1]], centers[args.axes[2]]]
    if focus:
        # centers = [args.mid_point[args.axes[0]], args.mid_point[args.axes[1]], args.mid_point[args.axes[2]] + 0.04]
        r = (int(np.sqrt(args.floor_dim)) - 1) * args.floor_unit_size / 2
        for ctr, dim in zip(centers, "xyz"):
            getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)

    if not "robot" in args.tool_type:
        ax.invert_yaxis()

    return outputs


# @profile
def render_anim(
    args,
    row_titles,
    state_seqs,
    attn_mask_pred=None,
    rels_pred=None,
    draw_set=["dough", "floor", "tool"],
    axis_off=False,
    target=None,
    views=[(90, -90), (0, -90), (45, -45)],
    fps=15,
    res="high",
    path="",
):
    n_frames = max([x.shape[0] for x in state_seqs])
    n_rows = len(row_titles)
    n_cols = len(views)

    fig_size = 12 if res == "high" else 3
    title_fontsize = 60 if res == "high" else 10
    fig, big_axes = plt.subplots(
        n_rows, 1, figsize=(fig_size * n_cols, fig_size * n_rows)
    )
    sm = cm.ScalarMappable(cmap="plasma")

    plot_info_dict = {}
    for i in range(n_rows):
        target_cur = target[i] if isinstance(target, list) else target

        if n_rows == 1:
            ax_cur = big_axes
        else:
            ax_cur = big_axes[i]

        ax_cur.set_title(row_titles[i], fontweight="semibold", fontsize=title_fontsize)
        ax_cur.axis("off")

        plot_info = []
        for j in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1, projection="3d")
            ax.view_init(*views[j])
            if attn_mask_pred is None:
                mask = None
            else:
                if j == n_cols - 1:
                    fig.colorbar(mappable=sm, ax=ax)
                mask = attn_mask_pred[i][0]

            if rels_pred is None:
                rels = None
            else:
                rels = rels_pred[i][0]

            outputs = visualize_points(
                ax,
                args,
                state_seqs[i][0],
                draw_set,
                target_cur,
                mask=mask,
                rels=rels,
                res=res,
                axis_off=axis_off,
            )

            plot_info.append((ax, outputs))

        plot_info_dict[row_titles[i]] = plot_info

    # plt.tight_layout()

    def update(step):
        outputs_all = []
        for i in range(n_rows):
            state = state_seqs[i]
            step_cur = min(step, state.shape[0] - 1)
            frame_cur = state[step_cur]
            if rels_pred is None:
                rels = None
            else:
                rels = rels_pred[i][step_cur]
            for j in range(n_cols):
                ax, outputs = plot_info_dict[row_titles[i]][j]
                draw_idx = 0
                if "dough" in draw_set:
                    outputs[draw_idx]._offsets3d = (
                        frame_cur[: args.n_particles, args.axes[0]],
                        frame_cur[: args.n_particles, args.axes[1]],
                        frame_cur[: args.n_particles, args.axes[2]],
                    )
                    if attn_mask_pred is not None:
                        mask = sm.to_rgba(attn_mask_pred[i][step_cur])
                        outputs[draw_idx].set(color=mask)
                    draw_idx += 1
                if "floor" in draw_set:
                    outputs[draw_idx]._offsets3d = (
                        frame_cur[
                            args.n_particles : args.n_particles + args.floor_dim,
                            args.axes[0],
                        ],
                        frame_cur[
                            args.n_particles : args.n_particles + args.floor_dim,
                            args.axes[1],
                        ],
                        frame_cur[
                            args.n_particles : args.n_particles + args.floor_dim,
                            args.axes[2],
                        ],
                    )
                    draw_idx += 1
                if "tool" in draw_set:
                    outputs[draw_idx]._offsets3d = (
                        frame_cur[args.n_particles + args.floor_dim :, args.axes[0]],
                        frame_cur[args.n_particles + args.floor_dim :, args.axes[1]],
                        frame_cur[args.n_particles + args.floor_dim :, args.axes[2]],
                    )
                    draw_idx += 1
                # if rels is not None and 'dough' in draw_set and 'tool' in draw_set:
                #     if args.full_repr:
                #         dough_color = [(0., 0., 1., 1.)] * args.n_particles
                #         tool_color = [(1., 0., 0., 1.)] * (frame_cur.shape[0] - args.n_particles - args.floor_dim)
                #         for k in range(rels.shape[0]):
                #             if rels[k][0] > 0 and rels[k][1] > 0:
                #                 for rel_idx in rels[k]:
                #                     if rel_idx < args.n_particles:
                #                         dough_color[rel_idx] = (0., 1., 0., 1.)
                #                     else:
                #                         tool_color[rel_idx - args.n_particles - args.floor_dim] = (0., 1., 0., 1.)
                #         outputs[0].set(edgecolor=dough_color)
                #         outputs[-1].set(edgecolor=tool_color)
                #     else:
                #         for k in range(rels.shape[0]):
                #             if rels[k][0] > 0 and rels[k][1] > 0:
                #                 outputs[draw_idx + k].set_data_3d(frame_cur[rels[k], args.axes[0]], frame_cur[rels[k], args.axes[1]],
                #                     frame_cur[rels[k], args.axes[2]])
                #             else:
                #                 outputs[draw_idx + k].set_data_3d([], [], [])

                outputs_all.extend(outputs)

        return outputs_all

    anim = animation.FuncAnimation(
        fig, update, frames=np.arange(0, n_frames + 2 * fps), blit=True
    )

    if len(path) > 0:
        anim.save(path, writer=animation.FFMpegWriter(fps=fps))
    else:
        plt.show()

    plt.close()


def render_frames(
    args,
    row_titles,
    state_seq,
    frame_list=[],
    axis_off=True,
    focus=True,
    draw_set=["dough", "floor", "tool"],
    target=None,
    views=[(90, -90), (0, -90), (45, -45)],
    res="high",
    path="",
    name="",
):
    n_frames = state_seq[0].shape[0]
    n_rows = len(row_titles)
    n_cols = len(views)

    fig_size = 12 if res == "high" else 3
    title_fontsize = 60 if res == "high" else 10
    fig, big_axes = plt.subplots(
        n_rows, 1, figsize=(fig_size * n_cols, fig_size * n_rows)
    )

    if len(frame_list) == 0:
        frame_list = range(n_frames)

    for frame in frame_list:
        for i in range(n_rows):
            state = state_seq[i]
            target_cur = target[i] if isinstance(target, list) else target
            focus_cur = focus[i] if isinstance(focus, list) else focus
            if n_rows == 1:
                big_axes.set_title(
                    row_titles[i], fontweight="semibold", fontsize=title_fontsize
                )
                big_axes.axis("off")
            else:
                big_axes[i].set_title(
                    row_titles[i], fontweight="semibold", fontsize=title_fontsize
                )
                big_axes[i].axis("off")

            for j in range(n_cols):
                ax = fig.add_subplot(
                    n_rows, n_cols, i * n_cols + j + 1, projection="3d"
                )
                ax.view_init(*views[j])
                visualize_points(
                    ax,
                    args,
                    state[frame],
                    draw_set,
                    target_cur,
                    axis_off=axis_off,
                    focus=focus_cur,
                    res=res,
                )

        # plt.tight_layout()

        if len(path) > 0:
            if len(name) == 0:
                plt.savefig(os.path.join(path, f"{str(frame).zfill(3)}.pdf"))
            else:
                plt.savefig(os.path.join(path, name))
        else:
            plt.show()

    plt.close()


def render_o3d(
    geometry_list,
    axis_off=False,
    focus=True,
    views=[(90, -90), (0, -90), (45, -45)],
    label_list=[],
    point_size_list=[],
    path="",
):
    n_rows = 2
    n_cols = 3

    fig, big_axes = plt.subplots(n_rows, 1, figsize=(12 * n_cols, 12 * n_rows))
    sm = cm.ScalarMappable(cmap="plasma")

    for i in range(n_rows):
        ax_cur = big_axes[i]

        title_fontsize = 60
        ax_cur.set_title("Test", fontweight="semibold", fontsize=title_fontsize)
        ax_cur.axis("off")

        for j in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1, projection="3d")
            # ax.computed_zorder = False
            ax.view_init(*views[j])
            if j == n_cols - 1:
                fig.colorbar(mappable=sm, ax=ax)

            for k in range(len(geometry_list)):
                type = geometry_list[k].get_geometry_type()
                # Point Cloud
                # if type == o3d.geometry.Geometry.Type.PointCloud:
                #     geometry.paint_uniform_color(pcd_color)
                # Triangle Mesh
                if type == o3d.geometry.Geometry.Type.TriangleMesh:
                    mf = pymeshfix.MeshFix(
                        np.asarray(geometry_list[k].vertices),
                        np.asarray(geometry_list[k].triangles),
                    )
                    # mf.repair()
                    mesh = mf.mesh
                    vertices = np.asarray(mesh.points)
                    triangles = np.asarray(mesh.faces)
                    ax.plot_trisurf(
                        vertices[:, 0],
                        vertices[:, 1],
                        triangles=triangles,
                        Z=vertices[:, 2],
                    )
                    # ax.set_aspect('equal')
                elif type == o3d.geometry.Geometry.Type.PointCloud:
                    particles = np.asarray(geometry_list[k].points)
                    colors = np.asarray(geometry_list[k].colors)
                    if len(point_size_list) > 0:
                        point_size = point_size_list[k]
                    else:
                        point_size = 160
                    if len(label_list) > 0:
                        label = label_list[k]
                        if "dough" in label:
                            ax.scatter(
                                particles[:, 0],
                                particles[:, 1],
                                particles[:, 2],
                                c="b",
                                s=point_size,
                                label=label,
                            )
                        elif "tool" in label:
                            ax.scatter(
                                particles[:, 0],
                                particles[:, 1],
                                particles[:, 2],
                                c="r",
                                alpha=0.2,
                                zorder=4.2,
                                s=point_size,
                                label=label,
                            )
                        else:
                            ax.scatter(
                                particles[:, 0],
                                particles[:, 1],
                                particles[:, 2],
                                c="yellowgreen",
                                zorder=4.1,
                                s=point_size,
                                label=label,
                            )
                    else:
                        label = None
                        ax.scatter(
                            particles[:, 0],
                            particles[:, 1],
                            particles[:, 2],
                            c=colors,
                            s=point_size,
                            label=label,
                        )
                else:
                    raise NotImplementedError

            if axis_off:
                ax.axis("off")

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            if len(label_list) > 0:
                ax.legend(fontsize=30, loc="upper right", bbox_to_anchor=(0.0, 0.0))

            # extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            # size = extents[:, 1] - extents[:, 0]
            centers = geometry_list[0].get_center()
            if focus:
                r = 0.05
                for ctr, dim in zip(centers, "xyz"):
                    getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)

    # plt.tight_layout()

    if len(path) > 0:
        plt.savefig(f'{path}_{datetime.now().strftime("%b-%d-%H:%M:%S")}.png')
    else:
        plt.show()

    plt.close()


def visualize_o3d(
    geometry_list,
    title="O3D",
    view_point=None,
    point_size=5,
    pcd_color=[0, 0, 0],
    mesh_color=[0.5, 0.5, 0.5],
    show_normal=False,
    show_frame=True,
    path="",
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(title)
    types = []

    for geometry in geometry_list:
        type = geometry.get_geometry_type()
        # Point Cloud
        # if type == o3d.geometry.Geometry.Type.PointCloud:
        #     geometry.paint_uniform_color(pcd_color)
        # Triangle Mesh
        if type == o3d.geometry.Geometry.Type.TriangleMesh:
            geometry.paint_uniform_color(mesh_color)
        types.append(type)

        vis.add_geometry(geometry)
        vis.update_geometry(geometry)

    vis.get_render_option().background_color = np.array([0.0, 0.0, 0.0])
    if show_frame:
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        vis.add_geometry(mesh)
        vis.update_geometry(mesh)

    if o3d.geometry.Geometry.Type.PointCloud in types:
        vis.get_render_option().point_size = point_size
        vis.get_render_option().point_show_normal = show_normal
    if o3d.geometry.Geometry.Type.TriangleMesh in types:
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().mesh_show_wireframe = True

    vis.poll_events()
    vis.update_renderer()

    if view_point is None:
        vis.get_view_control().set_front(np.array([0.305, -0.463, 0.832]))
        vis.get_view_control().set_lookat(np.array([0.4, -0.1, 0.0]))
        vis.get_view_control().set_up(np.array([-0.560, 0.620, 0.550]))
        vis.get_view_control().set_zoom(0.2)
    else:
        vis.get_view_control().set_front(view_point["front"])
        vis.get_view_control().set_lookat(view_point["lookat"])
        vis.get_view_control().set_up(view_point["up"])
        vis.get_view_control().set_zoom(view_point["zoom"])

    # cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    # path = os.path.join(cd, '..', 'figures', 'images', f'{title}_{datetime.now().strftime("%b-%d-%H:%M:%S")}.png')

    if len(path) > 0:
        vis.capture_screen_image(path, True)
        vis.destroy_window()
    else:
        vis.run()


def visualize_target(args, target_shape_name):
    target_frame_path = os.path.join(
        os.getcwd(),
        "target_shapes",
        target_shape_name,
        f'{target_shape_name.split("/")[-1]}.h5',
    )
    visualize_h5(args, target_frame_path)


def visualize_h5(args, file_path):
    hf = h5py.File(file_path, "r")
    data = []
    for i in range(len(args.data_names)):
        d = np.array(hf.get(args.data_names[i]))
        data.append(d)
    hf.close()
    target_shape = data[0][: args.n_particles, :]
    render_frames(args, ["H5"], [np.array([target_shape])], draw_set=["dough"])


def visualize_neighbors(args, particles, target, neighbors, path=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # red is the target and blue are the neighbors
    ax.scatter(
        particles[: args.n_particles, args.axes[0]],
        particles[: args.n_particles, args.axes[1]],
        particles[: args.n_particles, args.axes[2]],
        c="c",
        alpha=0.2,
        s=30,
    )
    ax.scatter(
        particles[args.n_particles :, args.axes[0]],
        particles[args.n_particles :, args.axes[1]],
        particles[args.n_particles :, args.axes[2]],
        c="r",
        alpha=0.2,
        s=30,
    )

    ax.scatter(
        particles[neighbors, args.axes[0]],
        particles[neighbors, args.axes[1]],
        particles[neighbors, args.axes[2]],
        c="b",
        s=60,
    )
    ax.scatter(
        particles[target, args.axes[0]],
        particles[target, args.axes[1]],
        particles[target, args.axes[2]],
        c="r",
        s=60,
    )

    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def plot_cm(test_set, y_true, y_pred, path=""):
    confusion_matrix = metrics.confusion_matrix(
        [test_set.classes[x] for x in y_true],
        [test_set.classes[x] for x in y_pred],
        labels=test_set.classes,
    )
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=test_set.classes
    )
    cm_display.plot(xticks_rotation="vertical")
    plt.gcf().set_size_inches(12, 12)
    plt.tight_layout()
    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def concat_images(imga, imgb, direction="h"):
    # combines two color image ndarrays side-by-side.
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]

    if direction == "h":
        max_height = np.max([ha, hb])
        total_width = wa + wb
        new_img = np.zeros(shape=(max_height, total_width, 3), dtype=imga.dtype)
        new_img[:ha, :wa] = imga
        new_img[:hb, wa : wa + wb] = imgb
    else:
        max_width = np.max([wa, wb])
        total_height = ha + hb
        new_img = np.zeros(shape=(total_height, max_width, 3), dtype=imga.dtype)
        new_img[:ha, :wa] = imga
        new_img[ha : ha + hb, :wb] = imgb

    return new_img


def concat_n_images(image_path_list, n_rows, n_cols):
    # combines N color images from a list of image paths
    row_images = []
    for i in range(n_rows):
        output = None
        for j in range(n_cols):
            idx = i * n_cols + j
            img_path = image_path_list[idx]
            img = plt.imread(img_path)[:, :, :3]
            if j == 0:
                output = img
            else:
                output = concat_images(output, img)
        row_images.append(output)

    output = row_images[0]
    # row_images.append(abs(row_images[1] - row_images[0]))
    for img in row_images[1:]:
        output = concat_images(output, img, direction="v")

    return output


def visualize_image_pred(img_paths, target, output, classes, path=""):
    concat_imgs = concat_n_images(img_paths, n_rows=2, n_cols=4)
    plt.imshow(concat_imgs)

    pred_str = ", ".join([classes[x] for x in output])
    plt.text(10, -30, f"prediction: {pred_str}", c="black")
    if target is not None:
        plt.text(10, -60, f"label: {classes[target]}", c="black")
    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def visualize_pcd_pred(
    row_titles,
    state_list,
    views=[(90, -90), (0, -90), (45, -45)],
    axis_off=False,
    res="low",
    path="",
):
    n_rows = len(row_titles)
    n_cols = len(views)

    fig_size = 12 if res == "high" else 3
    title_fontsize = 60 if res == "high" else 10
    point_size = 160 if res == "high" else 10
    fig, big_axes = plt.subplots(
        n_rows, 1, figsize=(fig_size * n_cols, fig_size * n_rows)
    )

    for i in range(n_rows):
        state = state_list[i]
        if n_rows == 1:
            big_axes.set_title(
                row_titles[i], fontweight="semibold", fontsize=title_fontsize
            )
            big_axes.axis("off")
        else:
            big_axes[i].set_title(
                row_titles[i], fontweight="semibold", fontsize=title_fontsize
            )
            big_axes[i].axis("off")

        for j in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1, projection="3d")
            ax.view_init(*views[j])
            state_colors = state[:, 3:6] if state.shape[1] > 3 else "b"
            ax.scatter(
                state[:, 0], state[:, 1], state[:, 2], c=state_colors, s=point_size
            )
            # ax.set_zlim(-0.075, 0.075)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            if axis_off:
                ax.axis("off")

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def visualize_tensor(tensor, path="", mode="RGB"):
    im = np.array(transforms.ToPILImage()(tensor[:3]))
    # if mode == 'HSV':
    #     im_array = np.array(im)
    #     im = cv.cvtColor(im_array, cv.COLOR_HSV2RGB)

    for i in range(3, tensor.shape[0], 3):
        # import pdb; pdb.set_trace()
        im_next = np.array(transforms.ToPILImage()(tensor[i : i + 3]))
        # if mode == 'HSV':
        #     im_next_array = np.array(im_next)
        #     im_next = cv.cvtColor(im_next_array, cv.COLOR_HSV2RGB)
        im = concat_images(im, im_next)
    plt.imshow(im)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Please specify the path of the pickle file!")
        exit()

    pkl_path = sys.argv[1]
    with open(pkl_path, "rb") as f:
        args_dict = pickle.load(f)
        anim_path = pkl_path.replace("_args", "").replace("pkl", "mp4")
        print(f"Rendering anim at {anim_path}...")
        render_anim(**args_dict, path=anim_path)


if __name__ == "__main__":
    main()
