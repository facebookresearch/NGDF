# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import os


def control_point_l1_loss(
    pred_control_points,
    gt_control_points,
    confidence=None,
    confidence_weight=None,
    mean_batch=True,
    device="cpu",
):
    """
    Computes the l1 loss between the predicted control points and the
    groundtruth control points on the gripper.
    """
    error = torch.sum(torch.abs(pred_control_points - gt_control_points), -1)
    error = torch.mean(error, -1)
    if confidence is not None:
        assert confidence_weight is not None
        error *= confidence
        confidence_term = (
            torch.mean(torch.log(torch.max(confidence, torch.tensor(1e-10).to(device))))
            * confidence_weight
        )

    if mean_batch:  # take mean over entire batch
        if confidence is None:
            return torch.mean(error)
        else:
            return torch.mean(error), -confidence_term
    else:
        return error


def get_control_point_tensor(batch_size, use_torch=True, device="cpu"):
    """
    Outputs a tensor of shape (batch_size x 6 x 3).
    use_tf: switches between outputing a tensor and outputing a numpy array.
    """
    control_points = np.load(
        f"{os.path.abspath(os.path.dirname(__file__))}/config/panda.npy"
    )[:, :3]
    control_points = [
        [0, 0, 0],
        [0, 0, 0],
        control_points[0, :],
        control_points[1, :],
        control_points[-2, :],
        control_points[-1, :],
    ]
    control_points = np.asarray(control_points, dtype=np.float32)
    control_points = np.tile(np.expand_dims(control_points, 0), [batch_size, 1, 1])

    if use_torch:
        return torch.tensor(control_points).to(device)

    return control_points


def transform_control_points(gt_grasps, batch_size, mode="tq", device="cpu"):
    """
    Transforms canonical points using gt_grasps.
    mode = 'tq' expects gt_grasps to have (batch_size x 7) where each
      element is catenation of translation and quaternion (wxyz) for each
      grasps.
    mode = 'rt': expects to have shape (batch_size x 4 x 4) where
      each element is 4x4 transformation matrix of each grasp.
    """
    assert mode == "tq" or mode == "rt", mode

    if not torch.is_tensor(gt_grasps):
        gt_grasps = torch.tensor(gt_grasps, dtype=torch.float32, device=device)

    grasp_shape = gt_grasps.shape
    if mode == "tq":
        assert len(grasp_shape) == 2, grasp_shape
        assert grasp_shape[-1] == 7, grasp_shape
        control_points = get_control_point_tensor(batch_size, device=device)
        num_control_points = control_points.shape[1]
        input_gt_grasps = gt_grasps

        gt_grasps = torch.unsqueeze(input_gt_grasps, 1).repeat(1, num_control_points, 1)

        gt_q = gt_grasps[:, :, 3:]
        gt_t = gt_grasps[:, :, :3]
        gt_control_points = qrot(gt_q, control_points)
        gt_control_points += gt_t

        return gt_control_points
    else:
        assert len(grasp_shape) == 3, grasp_shape
        assert grasp_shape[1] == 4 and grasp_shape[2] == 4, grasp_shape
        control_points = get_control_point_tensor(batch_size, device=device)
        shape = control_points.shape
        ones = torch.ones((shape[0], shape[1], 1), dtype=torch.float32, device=device)
        control_points = torch.cat((control_points, ones), -1)
        return torch.matmul(control_points, gt_grasps.permute(0, 2, 1))


def get_closest_idx_cp(query_cp, pos_cps):
    losses = control_point_l1_loss(query_cp, pos_cps, mean_batch=False)
    closest_cp_idx = torch.argmin(losses)
    cp_dist = losses[closest_cp_idx]

    per_pt_losses = torch.sum(torch.abs(query_cp - pos_cps), -1)
    cpd_dist = per_pt_losses[closest_cp_idx].max()
    return closest_cp_idx.item(), cpd_dist.item(), cp_dist.item()


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q (wxyz).
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)
