import os
import json
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def quat_to_rotate6D(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def norm(action, global_min, global_max):
    action = (action - global_min[None, :]) / (global_max[None, :] - global_min[None, :] + 1e-8)
    action = np.clip(action, 0, 1)
    return action


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_tsne(actions, save_path, perplexity=30, max_iter=1000, pca_dim=50):
    """
    actions: (N, D) array
    save_path: path to save the tsne plot
    """
    # if actions.shape[0] > 5000:   # keep only 5k points max
    #     idx = np.random.choice(actions.shape[0], 5000, replace=False)
    #     actions = actions[idx]

    pca_dim = min(pca_dim, actions.shape[1])  # avoid exceeding feature count
    # print(f"Running PCA -> {pca_dim} dims...")
    # actions_reduced = PCA(n_components=pca_dim).fit_transform(actions)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, init="pca", random_state=42)
    embedding = tsne.fit_transform(actions)

    plt.figure(figsize=(6,6))
    plt.scatter(embedding[:,0], embedding[:,1], s=2, alpha=0.5)
    plt.title("t-SNE after PCA")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def plot_pca(actions, save_path, pca_dim=2):
    """
    actions: (N, D) array
    save_path: path to save the PCA plot
    """
    print(f"Running PCA -> {pca_dim} dims...")
    actions_reduced = PCA(n_components=pca_dim).fit_transform(actions)

    plt.figure(figsize=(6,6))
    plt.scatter(actions_reduced[:,0], actions_reduced[:,1], s=2, alpha=0.5)
    plt.title("PCA Projection")
    plt.xlabel("PCA dim 1")
    plt.ylabel("PCA dim 2")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved PCA plot to {save_path}")



def collect_all_actions(metas_path, stats_file=None, field="action_seq", num_actions=10, model_type="continuous", num_bins=256, normalize=True):
    """
    Collect all data from the dataset at once instead of using the dataloader.
    Returns a big numpy array of values for the given field.
    """
    # Load meta info
    with open(metas_path, "r") as f:
        meta = json.load(f)
    datalist = meta["datalist"]
    dataset_name = meta["dataset_name"]
    try:
        stats = np.load(stats_file)
        global_mean = np.asarray(stats['mean'])
        global_std = np.asarray(stats['std'])
        global_min = np.asarray(stats['min'])
        global_max = np.asarray(stats['max'])
        print('global_mean', global_mean)
        print('global_std', global_std)
    except:
        print('stats_file is empty')
    collected = []
    xyz_all = []
    rot6_all = []

    if normalize:
        print('min max normalizing')
    for datapath in datalist:
        if not isinstance(datapath, str):
            datapath = datapath[0]

        with h5py.File(datapath, "r") as data:
            if dataset_name == "robotwin2_abs_ee":
                left_ee = data["endpose/left_endpose"][()]      # shape (T, 7)
                right_ee = data["endpose/right_endpose"][()]    # shape (T, 7)
                left_grip = data["endpose/left_gripper"][()]    # shape (T,)
                right_grip = data["endpose/right_gripper"][()]  # shape (T,)
                proprio = np.concatenate([
                    left_ee[:, :3],
                    quat_to_rotate6D(left_ee[:, 3:]),                        # (T,7)
                    left_grip[:, None],             # (T,1)
                    right_ee[:, :3],
                    quat_to_rotate6D(right_ee[:, 3:]),                       # (T,7)
                    right_grip[:, None]             # (T,1)
                ], axis=-1)
                left_xyz = left_ee[:, :3]
                right_xyz = right_ee[:, :3]
                if normalize: proprio = norm(proprio, global_min, global_max)

            elif dataset_name == "robotwin2_abs_qpos":
                left_joint = data["joint_action/left_arm"][()]
                right_joint = data["joint_action/right_arm"][()]
                left_grip = data["joint_action/left_gripper"][()]
                right_grip = data["joint_action/right_gripper"][()]
                proprio = np.concatenate([
                    left_joint,
                    left_grip[:, None],
                    right_joint,
                    right_grip[:, None]
                ], axis=-1)
                if normalize: proprio = norm(proprio, global_min, global_max)

            elif dataset_name == 'robotwin2_rel_ee':
                left_ee = data["endpose/left_endpose"][()]      # shape (T, 7)
                right_ee = data["endpose/right_endpose"][()]    # shape (T, 7)
                left_grip = data["endpose/left_gripper"][()]    # shape (T,)
                right_grip = data["endpose/right_gripper"][()]  # shape (T,)
                prorpio_seq = np.concatenate([
                    left_ee[:, :3],
                    quat_to_rotate6D(left_ee[:, 3:]),                        # (T,7)
                    left_grip[:, None],             # (T,1)
                    right_ee[:, :3],
                    quat_to_rotate6D(right_ee[:, 3:]),                       # (T,7)
                    right_grip[:, None]             # (T,1)
                ], axis=-1)
                left_delta_xyz = left_ee[1:, :3] - left_ee[:-1, :3]
                right_delta_xyz = right_ee[1:, :3] - right_ee[:-1, :3]
                left_delta_rot6d = cal_delta_rotate(left_ee[1:, 3:], left_ee[:-1, 3:])
                right_delta_rot6d = cal_delta_rotate(right_ee[1:, 3:], right_ee[:-1, 3:])
                ee_diff = np.concatenate([
                    left_delta_xyz,
                    left_delta_rot6d,
                    left_grip[1:, None],   # future gripper value
                    right_delta_xyz,
                    right_delta_rot6d,
                    right_grip[1:, None]
                ], axis=-1)
                proprio = ee_diff
            elif dataset_name == 'robotwin2_rel_qpos':
                left_joint = data["joint_action/left_arm"][()]      # shape (T, 7)
                right_joint = data["joint_action/right_arm"][()]    # shape (T, 7)
                left_grip = data["joint_action/left_gripper"][()]    # shape (T,)
                right_grip = data["joint_action/right_gripper"][()]  # shape (T,)
                prorpio_seq = np.concatenate([
                    left_joint,                        # (T,7)
                    left_grip[:, None],             # (T,1)
                    right_joint,                    # (T,7)
                    right_grip[:, None]             # (T,1)
                ], axis=-1)
                joint_diff = np.concatenate([
                    left_joint[1:] - left_joint[:-1],
                    left_grip[1:, None],  # use future value directly
                    right_joint[1:] - right_joint[:-1],
                    right_grip[1:, None]
                ], axis=-1)
                # if not self.discretize:
                #     action_seq = (joint_diff - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                # else:
                #     action_seq = joint_diff
                proprio = joint_diff
            else:
                raise NotImplementedError(f"Dataset type {dataset_name} not handled yet")
            action_seq = proprio[1:]
            # print('proprio', proprio.shape)
            collected.append(action_seq)
    collected = np.concatenate(collected, axis=0)  # (N, 3)
    # rot6_all = np.concatenate(rot6_all, axis=0)  # (N, 6)

    return collected


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

def plot_action_dim_distributions(actions, save_path, bandwidth=0.2):
    """
    Plot all action dimensions on a single graph with KDE curves.
    
    actions: (N, action_dim) numpy array
    save_path: path to save the single combined figure
    """

    print('actions', actions.shape)
    action_dim = actions.shape[1]

    plt.figure(figsize=(8, 6))

    for i in range(action_dim):
        if 'qpos' in save_path:
            if i == 6 or i == action_dim - 1 :
                print('continue')
                continue
        elif 'ee' in save_path:
            if i == 9 or i == action_dim - 1 :
                continue
        dim_values = actions[:, i][:, None]  # (N,1) for KDE

        # fit KDE
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(dim_values)
        x_grid = np.linspace(dim_values.min(), dim_values.max(), 500)[:, None]
        log_dens = kde.score_samples(x_grid)
        dens = np.exp(log_dens)

        # plot each dimension on the same figure
        plt.plot(x_grid[:, 0], dens, label=f"dim {i}")

    name = save_path.split('/')[-1].split('.')[0]
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("KDE of All Action Dimensions " + name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved combined KDE plot to {save_path}")



def plot_xyz_distribution(xyz, save_path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Distribution of XYZ")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved 3D xyz distribution to {save_path}")


def plot_rot6_on_sphere(rot6, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    v1 = rot6[:, 0:3]
    v2 = rot6[:, 3:6]

    for i, (vecs, name, color) in enumerate(
        [(v1, "rot6_first3", "red"), (v2, "rot6_last3", "green")]
    ):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2], s=1, c=color, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"Rot6 {name} vectors on unit sphere")
        save_path = os.path.join(save_dir, f"{name}_sphere.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved rot6 sphere plot to {save_path}")



# Example usage
stats_file = None
metas_path = "/home/fyc/EmpiricalStudyForVLA/datasets/meta_files/abs_ee_single_camera-50-10.jsonl"
save_folder = "distributions"
stats_file = "/home/fyc/EmpiricalStudyForVLA/datasets/meta_files/abs_ee_single_camera-50-10_global_stats_abs.npz"
collected = collect_all_actions(metas_path, stats_file)
file_name = metas_path.split('/')[-1].split('.')[0] + '.png'
print(file_name)
# plot_action_dim_distributions(collected, os.path.join(save_folder, file_name))
# plot_tsne(collected, save_folder)


save_folder = "tsne_results"
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, "tsne_"+file_name)

plot_tsne(collected, save_path, perplexity=30, max_iter=1000)

# Save 3D scatter for xyz
# plot_xyz_distribution(xyz_all, os.path.join(save_folder, "xyz_distribution.png"))

# Save sphere plots for rot6
# plot_rot6_on_sphere(rot6_all, save_folder)
