# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.offline as py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from acronym_tools import create_gripper_marker
from omegaconf import OmegaConf
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt

from ngdf.control_pts import transform_control_points
from ngdf.evaluate import evaluate, get_plotly_fig, pose_to_T
from ngdf.utils import get_input, load_mesh, wrist_to_tip

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")


def load_model(model_path):
    """Load shape embedding model"""
    model = vnn_occupancy_network.VNNOccNet(
        latent_dim=256,
        model_type="pointnet",
        return_features=False,
        sigmoid=True,
    ).cuda()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    return model


class Decoder(pl.LightningModule):
    def __init__(
        self,
        latent=None,
        dims=[],
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        input_in=(),
        weight_norm=False,
        xyz_in_all=None,
        lr=None,
        batch_size=8,
        use_tip=True,
        data_path="",
        wandb=False,
        max_test_samples=30,
        max_test_iter=10000,
        **kwargs,
    ):
        super().__init__()

        self.lr = lr
        self.latent = latent
        self.input_in = input_in
        self.batch_size = batch_size
        self.use_tip = use_tip
        self.data_path = data_path
        self.wandb = wandb
        self.max_test_samples = max_test_samples
        self.max_test_iter = max_test_iter

        self.save_hyperparameters()

        # input size
        ee_size = 7

        # output size
        out_size = 6

        # latent size
        latent_size = latent.size
        dims = [latent_size + ee_size] + list(dims) + [out_size]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        if self.latent.dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.input_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout

        self.shape_model = load_model(self.latent.run_path)

        self.training_losses = []

    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent.dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.input_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

            if layer == self.num_layers - 2:
                x = torch.nn.functional.softplus(x)

        return x

    def get_loss(self, y_hat, y):
        loss = F.l1_loss(y_hat, y)
        return loss

    def get_batch(self, batch):
        with torch.no_grad():
            if self.use_tip and not hasattr(self, "wrist_tip_T"):
                self.wrist_tip_T = wrist_to_tip(device=self.device).unsqueeze(0)
            query_Ts_orig, grasp_Ts_orig, info = batch
            if self.use_tip:
                query_Ts = query_Ts_orig @ self.wrist_tip_T
                grasp_Ts = grasp_Ts_orig @ self.wrist_tip_T
            else:
                query_Ts = query_Ts_orig
                grasp_Ts = grasp_Ts_orig

            # Get latent embedding
            pc = info["pc"][:, :, :3]
            shape_mi = {"point_cloud": pc}
            with torch.no_grad():
                latent = self.shape_model.extract_latent(shape_mi)  # 1 x B x 3
                latent = torch.reshape(latent, (latent.shape[0], -1))

            x_pose = get_input(query_Ts)

            x = torch.cat([x_pose, latent], axis=1)

            # Get output pose representation
            # control points code takes transforms with origin at wrist
            query_cps = transform_control_points(
                query_Ts_orig,
                batch_size=query_Ts_orig.shape[0],
                mode="rt",
                device=query_Ts_orig.device,
            )
            grasp_cps = transform_control_points(
                grasp_Ts_orig,
                batch_size=grasp_Ts_orig.shape[0],
                mode="rt",
                device=grasp_Ts_orig.device,
            )
            y = torch.sum(torch.abs(query_cps - grasp_cps), -1)

            info = {"pose": x_pose, "latent": latent, "obj": info["obj"]}
            return x, y, info

    def training_step(self, batch, batch_idx):
        x, y, _ = self.get_batch(batch)
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat, y)

        if batch_idx % 100 == 0:
            logging.info(
                f"TRAIN Epoch {self.current_epoch} Batch {batch_idx} Loss: {loss.item():.6f}"
            )

        self.log("loss/train", loss, logger=True, batch_size=self.batch_size)
        return {"loss": loss}

    def run_validation(self, batch, batch_idx):
        x, y, _ = self.get_batch(batch)
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat, y)
        if batch_idx % 100 == 0:
            logging.info(
                f"VAL Epoch {self.current_epoch} Batch {batch_idx} Loss: {loss.item():.6f}"
            )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            loss = self.run_validation(batch, batch_idx)
        elif dataloader_idx == 1:
            x, _, binfo = self.get_batch(batch)
            loss, traj, closest_T = evaluate(self, x, binfo, self.data_path)
            if batch_idx < 5 and self.wandb:
                self.log_visualization(binfo["obj"][0], traj, closest_T, batch_idx)
        self.log(f"loss/val", loss, logger=True, batch_size=self.batch_size)
        return {"loss": loss}

    def log_visualization(
        self,
        obj,
        traj,
        closest_T,
        batch_idx,
        use_wandb=True,
        plot_offline=False,
        fixed_skip=False,
        rotate_gripper=True,
        save_images=False,
        save_path=None,
    ):
        if self.use_tip:
            tip2wrist_T = np.linalg.inv(wrist_to_tip(device="cpu"))
        else:
            tip2wrist_T = np.eye(4)

        # Training occurs in pybullet wrist convention.
        # To visualize using the acronym create_gripper_marker function, need to switch back to acronym convention.
        # rotgrasp is the acronym convention, grasp is the pybullet convention.
        T_rotgrasp2grasp = pt.transform_from(
            pr.matrix_from_axis_angle([0, 0, 1, -np.pi / 2]), [0, 0, 0]
        )  # correct wrist rotation
        T_grasp2rotgrasp = np.linalg.inv(T_rotgrasp2grasp)

        query_iters = []
        if fixed_skip:
            skip = 10
        else:
            skip = 50 if len(traj) < 500 else len(traj) // 50
        traj.reverse()
        for i, (pose, dist) in enumerate(traj[::skip]):
            T = pose_to_T(pose, tip2wrist_T)
            if i == 0:
                query_iters.append(
                    create_gripper_marker(
                        color=[255, 0, 255], tube_radius=0.003, sections=6
                    ).apply_transform(T @ T_grasp2rotgrasp)
                )  # mark final point
            else:
                query_iters.append(
                    create_gripper_marker(
                        color=[0, 255, 255], sections=3
                    ).apply_transform(T @ T_grasp2rotgrasp)
                )

        # mesh_mctr and point cloud center are similar for our current objects; to visualize better we should account for translation to pctr.
        mesh, mesh_mctr_T = load_mesh(
            f"{self.data_path}/../grasps/{obj}.h5",
            mesh_root_dir=f"{self.data_path}/../",
            load_for_bullet=True,
        )
        grasp = create_gripper_marker(
            color=[0, 255, 0], tube_radius=0.003
        ).apply_transform(closest_T @ T_grasp2rotgrasp)

        # plotly
        print("process meshes for plotly")
        mesh_fig = get_plotly_fig(mesh)
        data = list(mesh_fig.data)
        grasp_fig = get_plotly_fig(grasp)
        data += list(grasp_fig.data)
        finalq_fig = get_plotly_fig(query_iters[0])
        data += list(finalq_fig.data)

        fig = go.Figure()
        fig.add_traces(data)
        fig.update_layout(
            coloraxis_showscale=False,
            scene=dict(
                xaxis=dict(visible=False, showticklabels=False),
                yaxis=dict(visible=False, showticklabels=False),
                zaxis=dict(visible=False, showticklabels=False),
                aspectmode="data",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.25, y=1.25, z=1.25),
                ),
            ),
        )
        if plot_offline:
            py.iplot(fig)
        if save_images:
            fig.write_image(
                f'{save_path}/levelsetfinal_{obj.split("_")[0]}_{batch_idx}.png',
                scale=1,
                width=3 * 640,
                height=3 * 480,
            )

        for i, query_iter in enumerate(query_iters[1:]):
            query_fig = get_plotly_fig(query_iter)
            query_fig["data"][0].update(opacity=0.7)
            data += list(query_fig.data)

        fig = go.Figure()
        fig.add_traces(data)
        fig.update_layout(
            coloraxis_showscale=False,
            scene=dict(
                xaxis=dict(visible=False, showticklabels=False),
                yaxis=dict(visible=False, showticklabels=False),
                zaxis=dict(visible=False, showticklabels=False),
                aspectmode="data",
            ),
        )

        if plot_offline:
            py.iplot(fig)
        if save_images:
            fig.write_image(
                f'{save_path}/levelsetpath_{obj.split("_")[0]}_{batch_idx}.png',
                scale=1,
                width=3 * 640,
                height=3 * 480,
            )
        if use_wandb:
            wandb_logger = (
                self.logger.experiment[2] if self.logger is not None else wandb
            )
            wandb_logger.log(
                {
                    f"3Dplot_{obj.split('_')[0]}_{batch_idx}": wandb.Html(
                        plotly.io.to_html(fig)
                    ),
                }
            )

        return mesh, mesh_mctr_T

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    # Testing
    cfg = OmegaConf.load("config/train.yaml")
    cfg.workers = 0
    decoder = Decoder(**cfg.net)

    cfg.aug_ratio = 1.0
    cfg.augment = "rot"
    cfg.net.batch_size = 2
    from dataset import GraspDataModule

    dm = GraspDataModule(cfg=cfg)
    dm.setup()
    loader = dm.train_dataloader()
    for batch in loader:
        query_Ts, grasp_Ts, info = batch
        x, y = decoder.get_batch(batch)
        pass
