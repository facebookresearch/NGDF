# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.offline as py
import pytorch_lightning.utilities.seed as seed_utils
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import torch
import wandb
from acronym_tools import create_gripper_marker
from omegaconf import OmegaConf

from ngdf.control_pts import get_closest_idx_cp, transform_control_points
from ngdf.dataset import GraspDataModule
from ngdf.utils import get_plotly_fig, load_mesh, wrist_to_tip


def optimize_floating_gripper(net, x, binfo, lr=0.001, loss_thresh=0.0001):
    max_iter = net.max_test_iter
    torch.set_grad_enabled(True)
    # optimize to minimize distance to grasp
    print("optimizing ")  # todo change to logger
    x_pose = binfo["pose"]
    latent = binfo["latent"]  # embedding
    traj = []
    x_pose = x_pose.detach().clone()
    x_pose.requires_grad = True
    opt = torch.optim.Adam([x_pose], lr=lr)
    for i in range(max_iter):
        # Get input for the current timestep
        x = torch.cat([x_pose, latent], axis=1)

        # Predict distance to closest grasp using network
        y_hat = net(x)

        # Save transform and distance for the current timestep
        traj.append((x_pose.detach().clone().squeeze(), y_hat))

        # Backprop to get grad x
        loss = y_hat.mean(axis=1)  # cpd
        # loss = torch.abs(loss) # Note this is because networks aren't currently constrained to be positive, will re-train
        if loss_thresh is not None and loss < loss_thresh:
            print(f"{i} loss {loss} below threshold {loss_thresh}, terminating early")
            break
        elif i % (max_iter // 100) == 0:
            print(f"{i}: {loss}")

        loss.backward()
        opt.step()
        opt.zero_grad()

        x_pose = opt.param_groups[0]["params"][0]

        qnorm = torch.linalg.norm(x_pose[:, 3:8])
        div = torch.ones_like(x_pose)
        div[:, 3:8] = qnorm
        x_pose = torch.div(x_pose, div)
    return x_pose, traj


def pose_to_T(pose, tip2wrist_T):
    """Convert pose in position+quaternion or position + 6D rotation to transform matrix"""
    T = pt.transform_from_pq(pose.detach().cpu().numpy()) @ tip2wrist_T
    return T


def get_closest_cp_dist(dset_grasp, obj, tip2wrist_T, traj):
    # Get closest grasp to final pose in dataset
    pos_Ts = dset_grasp[obj]["pos_Ts"]
    # convert pos Ts to pybullet wrist convention from acronym convention
    # query / end pose is already in pybullet convention
    T_rotgrasp2grasp = pt.transform_from(
        pr.matrix_from_axis_angle([0, 0, 1, -np.pi / 2]), [0, 0, 0]
    )
    pos_Ts = pos_Ts @ T_rotgrasp2grasp
    pos_cps = transform_control_points(
        pos_Ts, batch_size=pos_Ts.shape[0], mode="rt", device="cuda"
    )
    end_pose = traj[-1][0]
    end_T = pose_to_T(end_pose, tip2wrist_T)[np.newaxis, :]
    end_cp = transform_control_points(
        end_T, batch_size=end_T.shape[0], mode="rt", device="cuda"
    )
    pos_idx_cp, _, cp_dist = get_closest_idx_cp(end_cp, pos_cps)
    closest_T = pos_Ts[pos_idx_cp]
    print(f"Closest distance: {cp_dist:.3f}")
    return end_T, cp_dist, closest_T


def evaluate(net, x, binfo, data_path):
    start_time = time.time()
    dset_grasp = h5py.File(f"{data_path}/dataset.hdf5", "r")
    if net.use_tip:
        tip2wrist_T = np.linalg.inv(wrist_to_tip(device="cpu"))
    else:
        tip2wrist_T = np.eye(4)

    x_pose, traj = optimize_floating_gripper(net, x, binfo)
    obj = binfo["obj"][0]
    end_T, cp_dist, closest_T = get_closest_cp_dist(dset_grasp, obj, tip2wrist_T, traj)
    loss = cp_dist

    if False:  # debug visualization
        visualize_grasp_set(data_path, dset_grasp)

    elapsed_time = time.time() - start_time
    print(f"duration: {elapsed_time}")
    return loss, traj, closest_T


def visualize_grasp_set(data_path, dset_grasp):
    mesh, mesh_mctr_T = load_mesh(
        f"{data_path}/../grasps/{obj}.h5",
        mesh_root_dir=f"{data_path}/../",
        load_for_bullet=True,
    )
    pos_Ts = dset_grasp[obj]["pos_Ts"]
    grasps = [
        create_gripper_marker(color=[0, 255, 0], tube_radius=0.003).apply_transform(T)
        for T in pos_Ts
    ]

    mesh_fig = get_plotly_fig(mesh)
    data = list(mesh_fig.data)
    for grasp in grasps:
        grasp_fig = get_plotly_fig(grasp)
        data += list(grasp_fig.data)
    fig = go.Figure()
    fig.add_traces(data)
    fig.update_layout(coloraxis_showscale=False)
    import plotly.offline as py

    py.iplot(fig)


if __name__ == "__main__":
    from networks import Decoder

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", help="Full path to checkpoint")
    parser.add_argument("--data_path", help="Full path to pose dataset", default=None)
    parser.add_argument(
        "--pc_data_path", help="Full path to point cloud dataset", default=None
    )
    parser.add_argument(
        "--eval_objs", help="object classes to evaluate", nargs="*", default=[]
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--loss_thresh", type=float, default=None)  # 0.0001
    parser.add_argument("--max_iter", type=int, default=3000)
    parser.add_argument("--visualize", action="store_true", default=True)
    parser.add_argument("--max_samples", type=int, default=3)
    parser.add_argument("--final_viz_samples", type=int, default=20)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--plot_offline", action="store_true", default=False)
    parser.add_argument("--save_images", action="store_true", default=False)
    parser.add_argument("--eval_pybullet", action="store_true", default=False)
    args = parser.parse_args()

    # Load config with object names and grasps dataset
    cfg_path = Path(args.ckpt_path).parents[3] / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.data_root = cfg_path.parents[3]

    # override data path for intracategory
    if args.data_path is not None:
        cfg.net.data_path = args.data_path
    if args.pc_data_path is not None:
        cfg.pc_data_path = args.pc_data_path
    if args.eval_objs != []:
        cfg.obj_classes = args.eval_objs

    cfg.net.max_test_samples = args.max_samples
    cfg.net.max_test_iter = args.max_iter
    cfg.project_root = Path(os.path.abspath(__file__)).parents[1]

    # Load model from checkpoint
    net = Decoder.load_from_checkpoint(
        args.ckpt_path,
        latent=cfg.net.latent,
        data_path=cfg.net.data_path,
        max_test_samples=cfg.net.max_test_samples,
        max_test_iter=cfg.net.max_test_iter,
    )
    net.cuda()
    net.eval()

    # Set up dataloader
    cfg.net.batch_size = 1
    seed_utils.seed_everything(0)
    dm = GraspDataModule(cfg=cfg)
    dm.setup_test()
    loader = dm.test_dataloader()

    # Set up logging
    tstamp = datetime.now().strftime(
        f'%y-%m-%d_%H-%M-%S_{args.max_samples}_{args.max_iter}_thresh{args.loss_thresh}_{Path(args.ckpt_path).name.replace(".ckpt", "")}'
    )
    log_path = Path(cfg_path).parents[1] / "eval" / tstamp
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(log_path / "images")
    hf = h5py.File(log_path / "eval.hdf5", "w")
    dt = h5py.string_dtype(encoding="ascii")
    for obj in dm.grasps_test.objects:
        obj_grp = hf.create_group(obj)
        cp_dists = obj_grp.create_dataset("cp_dists", shape=(0,), maxshape=(None,))
        end_Ts = obj_grp.create_dataset(
            "end_Ts", shape=(0, 4, 4), maxshape=(None, 4, 4)
        )
        successes = obj_grp.create_dataset("successes", shape=(0,), maxshape=(None,))
        err_info = obj_grp.create_dataset(
            "err_info", shape=(0,), maxshape=(None,), dtype=dt
        )

    if args.use_wandb:
        wandb_cfg = {"args": vars(args), "net_cfg": cfg}
        wandb.init(
            id=tstamp,
            name=cfg.experiment,
            project=cfg.wdb_project,
            entity=cfg.wdb_entity,
            config=wandb_cfg,
            dir=str(log_path),
        )

    if args.eval_pybullet:
        from omg_bullet.envs.acronym_env import PandaAcronymEnv
        from omg_bullet.get_bullet_labels_mp import run_grasp

        env = PandaAcronymEnv(renders=False, egl_render=False, gravity=False)
    dset_grasp = h5py.File(f"{cfg.net.data_path}/dataset.hdf5", "r")
    tip2wrist_T = np.eye(4)
    for batch_idx, batch in enumerate(loader):
        samples_per_obj = [len(hf[obj]["cp_dists"]) for obj in dm.grasps_test.objects]
        if all(samples >= args.max_samples for samples in samples_per_obj):
            print("Max samples for all objects reached")
            break

        # get batch
        query_T, grasp_T, info = batch
        batch[0] = batch[0].cuda()
        batch[1] = batch[1].cuda()
        obj = info["obj"][0]
        hf_idx = len(hf[obj]["cp_dists"])
        if hf_idx >= args.max_samples:
            print(f"Skipping {obj}, max samples reached")
            continue
        hf[obj]["cp_dists"].resize(hf_idx + 1, axis=0)
        hf[obj]["end_Ts"].resize(hf_idx + 1, axis=0)
        hf[obj]["successes"].resize(hf_idx + 1, axis=0)
        hf[obj]["err_info"].resize(hf_idx + 1, axis=0)
        grasp_type = info["grasp_type"][0]
        assert grasp_type == "query"
        print(obj)

        batch[2]["pc"] = batch[2]["pc"].cuda()

        x, y, binfo = net.get_batch(batch)

        x_pose, traj = optimize_floating_gripper(
            net, x, binfo, lr=args.lr, loss_thresh=args.loss_thresh
        )
        end_T, cp_dist, closest_T = get_closest_cp_dist(
            dset_grasp, obj, tip2wrist_T, traj
        )

        hf[obj]["end_Ts"][hf_idx] = end_T
        hf[obj]["cp_dists"][hf_idx] = cp_dist

        # Visualize trajectory
        if args.visualize:  # debug visualization
            mesh, mctr_obj_T = net.log_visualization(
                obj,
                traj,
                closest_T,
                batch_idx,
                use_wandb=args.use_wandb,
                plot_offline=args.plot_offline,
                fixed_skip=True,
                save_images=args.save_images,
                save_path=str(log_path / "images"),
            )

        if args.eval_pybullet:
            mesh_root = str(Path(net.data_path).parent)
            success, err_info = run_grasp(
                env, mesh, mctr_obj_T, mesh_root, obj, end_T[0]
            )
            hf[obj]["successes"][hf_idx] = success
            hf[obj]["err_info"][hf_idx] = err_info

        # Save data and compute metrics
        dict_obj = dict([(obj, pd.Series(hf[obj]["cp_dists"])) for obj in hf.keys()])
        df_obj = pd.DataFrame.from_dict(dict_obj)
        df_obj_stats = pd.DataFrame(
            df_obj.mean().round(4).astype(str)
            + " +/- "
            + df_obj.std().round(3).astype(str)
        )
        all_dists = [v for obj in hf.keys() for v in hf[obj]["cp_dists"]]
        df_all = pd.DataFrame(all_dists)
        df_all_stats = pd.DataFrame(
            df_all.mean().round(4).astype(str)
            + " +/- "
            + df_all.std().round(3).astype(str)
        )
        dict_success = dict(
            [(obj, pd.Series(hf[obj]["successes"])) for obj in hf.keys()]
        )
        df_success = pd.DataFrame.from_dict(dict_success)
        df_success_stats = pd.DataFrame(
            df_success.mean().round(4).astype(str)
            + " +/- "
            + df_obj.std().round(3).astype(str)
        )
        with open(log_path / "df_obj_stats.md", "w") as f:
            f.write(df_obj_stats.to_markdown())
        with open(log_path / "df_all_stats.md", "w") as f:
            f.write(df_all_stats.to_markdown())
        with open(log_path / "df_success_stats.md", "w") as f:
            f.write(df_success_stats.to_markdown())
        print(df_obj_stats.to_markdown())
        print(df_all_stats.to_markdown())
        print(df_success_stats.to_markdown())

    # Visualize end_Ts for each object
    if args.visualize:
        T_rotgrasp2grasp = pt.transform_from(
            pr.matrix_from_axis_angle([0, 0, 1, -np.pi / 2]), [0, 0, 0]
        )  # correct wrist rotation
        T_grasp2rotgrasp = np.linalg.inv(T_rotgrasp2grasp)
        for obj in dm.grasps_test.objects:
            end_Ts = np.array(hf[obj]["end_Ts"])
            end_idxs = np.random.choice(
                range(len(end_Ts)),
                size=len(end_Ts)
                if len(end_Ts) < args.final_viz_samples
                else args.final_viz_samples,
            )
            end_Ts = end_Ts[end_idxs]
            mesh, mesh_mctr_T = load_mesh(
                f"{cfg.net.data_path}/../grasps/{obj}.h5",
                mesh_root_dir=f"{cfg.net.data_path}/../",
                load_for_bullet=True,
            )
            # note: mctr frame instead of pctr frame
            grippers = [
                create_gripper_marker(
                    color=[0, 255, 255], tube_radius=0.003, sections=4
                ).apply_transform(end_T @ T_grasp2rotgrasp)
                for end_T in end_Ts
            ]
            mesh_fig = get_plotly_fig(mesh)
            data = list(mesh_fig.data)
            for gripper in grippers:
                gripper_fig = get_plotly_fig(gripper)
                data += list(gripper_fig.data)
            fig = go.Figure()
            fig.add_traces(data)
            fig.update_layout(coloraxis_showscale=False)
            if args.plot_offline:
                py.iplot(fig)
            if args.use_wandb:
                wandb.log(
                    {
                        f"3Dfinal_{obj.split('_')[0]}": wandb.Html(
                            plotly.io.to_html(fig)
                        ),
                    }
                )
