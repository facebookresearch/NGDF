# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from copy import deepcopy

import h5py
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from torch.utils.data import DataLoader, Dataset, random_split

from ngdf.utils import load_mesh


class GraspDataset(Dataset):
    def __init__(self, config, dset_grasp, objects, ids, pc_dict={}, augment=False):
        self.cfg = config
        self.dset_grasp = dset_grasp
        self.objects = objects
        self.ids = ids
        self.augment = augment

        self.mctr_mesh_Ts = {}
        self.model = vnn_occupancy_network.VNNOccNet(
            latent_dim=256,
            model_type="pointnet",
            return_features=False,
            sigmoid=True,
        )
        self.model.load_state_dict(
            torch.load(self.cfg.net.latent.run_path), strict=False
        )
        self.model.eval()

        self.pc_dict = pc_dict
        for obj in self.objects:
            mesh, mesh_mctr_T = load_mesh(
                f"{self.cfg.net.data_path}/../grasps/{obj}.h5",
                mesh_root_dir=f"{self.cfg.net.data_path}/../",
                load_for_bullet=True,
            )
            self.mctr_mesh_Ts[obj] = np.linalg.inv(mesh_mctr_T)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample = self.ids[index]
        obj_id, grasp_type, grasp_idx = sample
        obj = self.objects[obj_id]

        # get query pose in mesh centroid frame
        mctr2ee_T = self.dset_grasp[obj][f"{grasp_type}_Ts"][
            grasp_idx
        ]  # centroid frame to query pose

        # get closest grasp in centroid frame
        if grasp_type == "pos":
            mctr2grasp_T = deepcopy(mctr2ee_T)
        else:
            closest_idx = self.dset_grasp[obj][f"pos_idxs_{self.cfg.dist_func}"][
                grasp_idx
            ]
            mctr2grasp_T = self.dset_grasp[obj]["pos_Ts"][closest_idx]

        info = {"obj": obj, "grasp_type": grasp_type}

        # Train network with grasps in pybullet wrist convention (grasp), transform from acronym convention (rotgrasp)
        T_rotgrasp2grasp = pt.transform_from(
            pr.matrix_from_axis_angle([0, 0, 1, -np.pi / 2]), [0, 0, 0]
        )  # correct wrist rotation
        p_mctr2ee_T = mctr2ee_T @ T_rotgrasp2grasp
        p_mctr2grasp_T = mctr2grasp_T @ T_rotgrasp2grasp

        # get point cloud
        if not hasattr(self, "dset_pc"):
            self.dset_pc = h5py.File(f"{self.cfg.pc_data_path}/dataset.hdf5", "r")
        rnd_idx = random.choice(self.pc_dict[obj])
        sample = self.dset_pc[obj][rnd_idx]

        # Get point cloud in 'mesh' frame (aka 'obj' frame)
        pc_world_fps = np.array(sample["pc_world_fps_t"])
        world_mesh_T = np.linalg.inv(np.array(sample["T_world_obj"]))
        pc_mesh_fps = (world_mesh_T @ pc_world_fps.T).T

        # Get grasps in 'mesh' frame (aka 'obj' frame)
        mctr_mesh_T = self.mctr_mesh_Ts[obj]
        mesh2ee_T = mctr_mesh_T @ p_mctr2ee_T
        mesh2grasp_T = mctr_mesh_T @ p_mctr2grasp_T

        # get point cloud in point cloud center 'pctr' frame
        mesh_pctr_T = np.eye(4)
        mesh_pctr_T[:3, 3] -= pc_mesh_fps.mean(axis=0)[:3]
        info["mesh_pctr_T"] = mesh_pctr_T
        pc_pctr_fps = (mesh_pctr_T @ pc_mesh_fps.T).T

        # get grasp in point cloud center 'pctr' frame
        pctr2ee_T = mesh_pctr_T @ mesh2ee_T
        pctr2grasp_T = mesh_pctr_T @ mesh2grasp_T

        # rename for convenience
        pc = pc_pctr_fps
        query_T = pctr2ee_T
        grasp_T = pctr2grasp_T

        # random augmentation of point cloud and grasp
        aug_this_sample = np.random.uniform(0, 1) < self.cfg.aug_ratio
        if self.augment and aug_this_sample:
            # Note difference between self.augment and self.cfg.augment
            if self.cfg.augment == "rot":
                rand_rot = np.random.uniform(-np.pi, np.pi, size=3)
            T_rot = np.eye(4)
            T_rot[:3, :3] = pr.matrix_from_euler_xyz(rand_rot)

            pc = (T_rot @ pc.T).T
            query_T = T_rot @ query_T
            grasp_T = T_rot @ grasp_T

        info["pc"] = pc.astype(np.float32)

        # gripper wrist is the origin of the poses
        return (
            torch.tensor(query_T, dtype=torch.float32),
            torch.tensor(grasp_T, dtype=torch.float32),
            info,
        )


class PointCloudDataset(Dataset):
    """Split into train and val within dataset
    so each can be assigned to train val grasp dataset
    """

    def __init__(self, config):
        self.cfg = config
        self.data_path = config.net.data_path

        self.dset_pc = h5py.File(f"{self.cfg.pc_data_path}/dataset.hdf5", "r")
        self.objects = [
            x
            for x in list(self.dset_pc.keys())
            if x.split("_")[0] in self.cfg.obj_classes
        ]

    def get_dicts(self):
        train_pc_dict = {}
        val_pc_dict = {}
        for obj in self.objects:
            ids = list(self.dset_pc[obj].keys())
            if self.cfg.n_pc_data_per_obj != -1:
                ids = ids[: self.cfg.n_pc_data_per_obj]

            n_val = int(len(ids) * self.cfg.pc_val_ratio)
            n_train = len(ids) - n_val
            ids_train, ids_val = random_split(ids, [n_train, n_val])
            train_pc_dict[obj] = ids_train
            val_pc_dict[obj] = ids_val
        return train_pc_dict, val_pc_dict


class GraspDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def collect_grasps(self, dset_grasp, objects, test=False):
        """Get tuples of object i, grasp type, and query index.
        Ground truth is the same as the query index for the given grasp type"""
        ids = []
        for i, obj in enumerate(objects):
            if test:
                ids += [
                    (i, "query", idx) for idx in range(len(dset_grasp[obj]["eval_Ts"]))
                ]
            else:
                ids += [
                    (i, "pos", idx) for idx in range(len(dset_grasp[obj]["pos_Ts"]))
                ]
                ids += [
                    (i, "query", idx) for idx in range(len(dset_grasp[obj]["query_Ts"]))
                ]
        return ids

    def setup(self, stage=None):
        # get ids of train/val split
        dset_grasp = h5py.File(f"{self.cfg.net.data_path}/dataset.hdf5", "r")
        objects = [
            x
            for x in list(dset_grasp.keys())
            if x.split("_")[0] in self.cfg.obj_classes
        ]

        ids = self.collect_grasps(dset_grasp, objects)
        random.shuffle(ids)

        if self.cfg.n_grasp_data != -1:  # Truncate grasp set
            ids = ids[: self.cfg.n_grasp_data]

        n_val = int(len(ids) * (self.cfg.grasps_val_ratio))
        n_train = len(ids) - n_val
        train_ids, val_ids = random_split(ids, [n_train, n_val])

        pc_dset = PointCloudDataset(self.cfg)
        train_pc_dict, val_pc_dict = pc_dset.get_dicts()
        self.grasps_train = GraspDataset(
            self.cfg,
            dset_grasp,
            objects,
            train_ids,
            train_pc_dict,
            augment=(self.cfg.augment != "off"),
        )
        self.grasps_val = GraspDataset(
            self.cfg, dset_grasp, objects, val_ids, val_pc_dict
        )
        print(
            f"Train: {len(self.grasps_train)} Batches: {len(self.grasps_train) / self.cfg.net.batch_size}"
        )
        print(
            f"Val: {len(self.grasps_val)} Batches: {len(self.grasps_val) / self.cfg.net.batch_size}"
        )
        self.setup_test()

    def setup_test(self):
        dset_grasp = h5py.File(f"{self.cfg.net.data_path}/dataset.hdf5", "r")
        objects = [x for x in list(dset_grasp.keys()) if x in self.cfg.obj_classes]

        test_ids = self.collect_grasps(dset_grasp, objects, test=True)
        test_ids = test_ids[: self.cfg.net.max_test_samples]
        pc_dset = PointCloudDataset(self.cfg)
        _, val_pc_dict = pc_dset.get_dicts()
        self.grasps_test = GraspDataset(
            self.cfg, dset_grasp, objects, test_ids, val_pc_dict
        )
        print(f"Test: {len(self.grasps_test)}")

    def train_dataloader(self):
        return DataLoader(
            self.grasps_train,
            batch_size=self.cfg.net.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                self.grasps_val,
                batch_size=self.cfg.net.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=self.cfg.workers,
                pin_memory=True,
            ),
        ]

    def test_dataloader(self):
        return DataLoader(
            self.grasps_test,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=self.cfg.workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    # Testing
    cfg = OmegaConf.load("config/train.yaml")
    cfg.workers = 0
    cfg.aug_ratio = 1.0
    cfg.augment = "rot"
    cfg.net.batch_size = 2
    dm = GraspDataModule(cfg=cfg)
    dm.setup()
    loader = dm.train_dataloader()

    for batch in loader:
        _ = batch
        pass
