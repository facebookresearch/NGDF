# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import numpy as np
import plotly.figure_factory as ff
import theseus as th
import torch
from acronym_tools import load_mesh as ac_load_mesh


def load_grasps(filename):
    """Load transformations and qualities of grasps from a JSON file from the dataset.
    Similar to the one in acronym_tools but also returns bullet_success

    Args:
        filename (str): HDF5 or JSON file name.

    Returns:
        np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
        np.ndarray: List of binary values indicating grasp success in simulation.
    """
    if filename.endswith(".h5"):
        data = h5py.File(filename, "r")
        T = np.array(data["grasps/transforms"])
        flex_success = np.array(data["grasps/qualities/flex/object_in_gripper"])
        try:
            bullet_success = np.array(data["grasps/qualities/bullet/object_in_gripper"])
        except KeyError as e:
            print(e)
            bullet_success = None
    else:
        raise RuntimeError("Unknown file ending:", filename)
    return T, flex_success, bullet_success


def load_mesh(h5, mesh_root_dir, load_for_bullet=False):
    # mesh loaded is scaled but not centered
    obj_mesh = ac_load_mesh(
        h5, mesh_root_dir=mesh_root_dir, load_for_bullet=load_for_bullet
    )
    T_obj_ctr = np.eye(4)
    T_obj_ctr[:3, 3] = -obj_mesh.centroid
    obj_mesh = obj_mesh.apply_transform(T_obj_ctr)
    return obj_mesh, T_obj_ctr


def transform_to_pq(T):
    """Convert batch transform to position+quaternion (wxyz) using theseus"""
    with torch.no_grad():
        assert T.ndim == 3
        xyzs = T[:, :3, 3]
        rots = th.SO3(tensor=T[:, :3, :3], strict=False).to_quaternion()  # wxyz
        x = torch.cat([xyzs, rots], axis=1)
        return x


def scale_logmap(log_map, scale=1.0):
    """Scale logmap

    Args:
        log_map (_type_): logmap to be scaled
        scale (float, optional): scale for translation component of logmap. Defaults to 1.0.

    Returns:
        _type_: scaled logmap
    """
    device = log_map.device
    scaled_log_map = log_map * torch.tensor(
        [scale, scale, scale, 1.0, 1.0, 1.0], device=device
    )
    return scaled_log_map


def get_input(query_T):
    """Get input in in_type representation"""
    x = transform_to_pq(query_T)
    return x


def wrist_to_tip(device="cpu", dtype=torch.float32):
    T_tip = torch.eye(4, device=device, dtype=dtype)
    T_tip[2, 3] = 0.10527314245700836
    return T_tip


def get_plotly_fig(mesh, plot_edges=True, opacity=1.0):
    x, y, z = mesh.vertices.T
    size = (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)).T
    fig = ff.create_trisurf(
        x=x,
        y=y,
        z=z,
        simplices=mesh.faces,
        colormap=[
            f"rgb({x},{y},{z})" for x, y, z in mesh.visual.face_colors[:, :3].tolist()
        ],
        aspectratio={"x": 1, "y": 1, "z": 1},
        show_colorbar=False,
        plot_edges=plot_edges,
    )
    if opacity != 1.0:
        fig["data"][0].update(opacity=opacity)
    return fig
