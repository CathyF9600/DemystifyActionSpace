import os
import json
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy.stats import entropy, multivariate_normal
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import argparse
import json_numpy
import requests
import cv2
from PIL import Image
import base64


def compute_per_dim_entropy(data, bins=30):
    """Histogram-based entropy per dimension."""
    entropies = []
    for i in range(data.shape[1]):
        hist, _ = np.histogram(data[:, i], bins=bins, density=True)
        hist = hist[hist > 0]
        entropies.append(entropy(hist))
    return np.array(entropies)


def plot_entropy_bar(ent_ee, ent_qpos, data_type: str, save_path="entropy_bar.png"):
    """Sorted bar chart comparing per-dimension entropies for ee and qpos."""
    dims = np.arange(1, len(ent_ee) + 1)

    # Sort indices by EE entropy (descending)
    sorted_idx = np.argsort(ent_ee)[::-1]
    ent_ee_sorted = np.array(ent_ee)[sorted_idx]
    sorted_idx = np.argsort(ent_qpos)[::-1]

    ent_qpos_sorted = np.array(ent_qpos)[sorted_idx]
    dims_sorted = dims[sorted_idx]

    width = 0.35
    x = np.arange(len(ent_ee))  # new sequential positions for sorted bars

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, ent_ee_sorted, width, label="EE")
    plt.bar(x + width/2, ent_qpos_sorted, width, label="Qpos")

    plt.xticks(x, dims_sorted)  # tick labels = original dimension indices
    plt.xlabel("Dimension (sorted by entropy)")
    plt.ylabel("Entropy")
    title = "Per-Dimension Entropy for " + data_type + " ee and qpos (Sorted)"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved bar chart to {save_path}")

def plot_entropy_heatmap(ent_ee, ent_qpos, save_path="entropy_heatmap.png"):
    """Heatmap visualization of entropy values, sorted by EE entropy."""
    # Sort indices by EE entropy (descending)
    sorted_idx = np.argsort(ent_ee)[::-1]
    ent_ee_sorted = np.array(ent_ee)[sorted_idx]
    sorted_idx = np.argsort(ent_qpos)[::-1]
    ent_qpos_sorted = np.array(ent_qpos)[sorted_idx]

    data = np.vstack([ent_ee_sorted, ent_qpos_sorted])
    labels = ["EE", "Qpos"]

    plt.figure(figsize=(12, 2))
    sns.heatmap(
        data,
        annot=True, fmt=".2f", cmap="viridis", cbar=True,
        xticklabels=sorted_idx + 1,  # original dimension indices (1-based)
        yticklabels=labels
    )
    plt.xlabel("Dimension (sorted by EE entropy)")
    plt.ylabel("Variable Set")
    plt.title("Entropy Heatmap (Sorted)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved heatmap to {save_path}")


def compute_entropy(data, bins=30):
    """Compute entropy per dimension and joint entropy (histogram + Gaussian estimate)."""
    data = np.asarray(data)
    ent_per_dim = []
    for i in range(data.shape[1]):
        hist, _ = np.histogram(data[:, i], bins=bins, density=True)
        hist = hist[hist > 0]
        ent_per_dim.append(entropy(hist))

    cov = np.cov(data.T)
    d = data.shape[1]
    det = np.linalg.det(cov)
    if det <= 0:
        gauss_entropy = np.nan
    else:
        gauss_entropy = 0.5 * np.log((2 * np.pi * np.e) ** d * det)

    return ent_per_dim, gauss_entropy


def decode_image_from_bytes(camera_rgb_image):
    if isinstance(camera_rgb_image, (bytes, bytearray)): camera_rgb_image = np.frombuffer(camera_rgb_image, dtype=np.uint8)
    rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
    if rgb is None: 
        rgb = np.frombuffer(camera_rgb_image, dtype=np.uint8) 
        if rgb.size == 2764800: 
            rgb = rgb.reshape(720, 1280, 3) 
        elif rgb.size == 921600: 
            rgb = rgb.reshape(480, 640, 3)
    return Image.fromarray(rgb)

def rot6_to_matrix(rot6):
    """
    rot6: shape (..., 6)
    return: rotation matrix (..., 3, 3)
    """
    a1 = rot6[..., 0:3]
    a2 = rot6[..., 3:6]

    # Gram-Schmidt 正交化
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)

def rot6_to_euler(rot6, seq="xyz"):
    Rmat = rot6_to_matrix(rot6)
    r = R.from_matrix(Rmat)
    ret= np.squeeze(r.as_euler(seq, degrees=True))  # (..., 3)
    print('ret', ret.shape)
    return ret

def quat_to_rotate6D(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def minmax_normalize(action, global_min, global_max):
    action = (action - global_min[None, :]) / (global_max[None, :] - global_min[None, :] + 1e-8)
    action = np.clip(action, 0, 1)
    return action

def meanstd_normalize(action, global_mean, global_std):
    # action = (action - global_min[None, :]) / (global_max[None, :] - global_min[None, :] + 1e-8)
    action = (action - global_mean[None, :]) / (global_std[None, :] + 1e-8)
    # action = np.clip(action, 0, 1)
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


def cal_delta_rotate(q1, q2):
    q1 = R.from_quat(q1)
    q2 = R.from_quat(q2)
    del_rotate = q1 * q2.inv()
    return del_rotate.as_matrix()[..., :, :2].reshape(q1.as_quat().shape[:-1] + (6,))


def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return np.mean(loss, axis=(1, 2))  # average over horizon (10) and action dim (20)

def plot_loss(loss):
    return

def get_loss(pred_actions, gt):
    # input: action: (N-1, 10, 20), gt (N, 20) 
    # get loss between actions[i-1, j, 0] and gt[i, :] (the prediction made by (i-1)th sequence's j-th action for j in [i, i+10]
    # output: loss: dict - key: action (20,), value: loss (np.float)
    N_minus_1, horizon, dim = pred_actions.shape

    # build gt chunks, but truncate at the end
    gt_chunks = np.zeros_like(pred_actions)  # placeholder
    mask = np.zeros((N_minus_1, horizon), dtype=bool)

    for i in range(N_minus_1):
        end = min(N_minus_1+1, i+1+horizon)
        gt_chunk = gt[i+1:end]
        gt_chunks[i, :len(gt_chunk)] = gt_chunk
        mask[i, :len(gt_chunk)] = True

    # compute huber loss for all
    losses_all = huber_loss_vectorized(pred_actions, gt_chunks, delta)  # (N-1,)

    # mask out invalid positions by recomputing only where mask=True
    losses = {}
    for i in range(N_minus_1):
        valid = mask[i]
        if np.any(valid):
            losses[i] = float(
                huber_loss(
                    pred_actions[i:i+1, valid],
                    gt_chunks[i:i+1, valid],
                    delta
                )[0]
            )
        else:
            losses[i] = None  # no valid gt
    
    return losses


def collect_all_actions(metas_path, stats_file=None, field="action_seq", num_actions=10, model_type="continuous", num_bins=256, normalize=True, compute_loss=False, args=None):
    """
    Collect all data from the dataset at once instead of using the dataloader.
    Returns a big numpy array of values for the given field.
    """
    if compute_loss:
        url = f"http://{args.host}:{args.port}/act"
        print('url', url)

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
    data_type = None
    if normalize:
        # print('min max normalizing')
        print('min', global_min, 'max', global_max)
    for datapath in datalist:
        # get parent directory name
        folder_name = os.path.basename(os.path.dirname(os.path.dirname(datapath)))
        # replace underscores with spaces
        task_name = folder_name.replace("_", " ")

        print('task_name', task_name) 

        if not isinstance(datapath, str):
            datapath = datapath[0]

        with h5py.File(datapath, "r") as data:
            images = data['observation/head_camera/rgb'][()] 

            if dataset_name == "robotwin2_abs_ee":
                data_type = "abs"
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
                if normalize: proprio = meanstd_normalize(proprio, global_mean, global_std)

            elif dataset_name == "robotwin2_abs_qpos":
                data_type = "abs"
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
                if normalize: proprio = meanstd_normalize(proprio, global_mean, global_std)

            elif dataset_name == 'robotwin2_rel_ee':
                data_type = "rel"
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
                if normalize:     
                    # proprio = meanstd_normalize(proprio, global_min, global_max)
                    proprio = meanstd_normalize(proprio, global_mean, global_std)

            elif dataset_name == 'robotwin2_rel_qpos':
                data_type = "rel"
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
                if normalize: 
                    # proprio = meanstd_normalize(proprio, global_min, global_max)
                    proprio = meanstd_normalize(proprio, global_mean, global_std)

            else:
                raise NotImplementedError(f"Dataset type {dataset_name} not handled yet")

            print('proprio', proprio.shape)
            action_seq = proprio[1:]
            collected.append(action_seq)

            if compute_loss:
                for idx in range(proprio.shape[0]): # (139, 20)
                    print("images[idx]", type(images[idx]))
                    image_input = np.array(decode_image_from_bytes(images[idx]))
                    print('image_input type', type(image_input))
                    proprio_input = proprio[0]
                    print('proprio_input', proprio_input.shape)
                    # instruction = args["task_name"].replace('_', ' ') # np.random.choice(results[0][instruction_type])
                    query = {
                        # "proprio": json_numpy.dumps(np.zeros_like(self.proprio)), #.reshape(1,-1),  # (1, 14)
                        "proprio": json_numpy.dumps(proprio_input),
                        "language_instruction": task_name,
                        "image0": json_numpy.dumps(image_input),
                        "data_type": data_type
                        # "image1": json_numpy.dumps(left_view),
                        # "image2": json_numpy.dumps(right_view)
                    }

                    response = requests.post(url, json=query).json()
                    noise_action = np.array(response['action']).squeeze(0)
                    print('noise_action', noise_action.shape)
                    loss = get_loss(noise_action)
                    plot_loss(loss)
            
    collected = np.concatenate(collected, axis=0)  # (N, 3)
    # rot6_all = np.concatenate(rot6_all, axis=0)  # (N, 6)
    # Suppose arr is your array of shape (N, 14)
    mins = collected.min(axis=0)  # shape (14,)
    maxs = collected.max(axis=0)  # shape (14,)

    print("Min values per feature:", mins)
    print("Max values per feature:", maxs)
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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_qpos_eef_correspondence(qpos, ee, save_path=None):
    """
    qpos_all: (N, D1)
    eef_all: (N, D2)
    save_path: optional, where to save the figure
    """
    # print('qpos_all', qpos_all.shape, eef_all.shape)

    N = qpos.shape[0]
    print('N', N)
    # 拼在一起做 PCA，保证投影空间一致
    combined = np.concatenate([qpos, ee], axis=0)  # 按行拼接，shape为(2N, max(D1,D2))
    pca = PCA(n_components=2)
    pca.fit(combined)  # 用所有数据拟合PCA模型
    
    # 使用同一个PCA模型分别转换qpos和ee
    qpos_2d = pca.transform(qpos)  # (N, 2)
    ee_2d = pca.transform(ee)      # (N, 2)
    print('qpos_2d.shape', qpos_2d.shape, ee_2d.shape)
    # 画散点
    plt.figure(figsize=(6,6))
    plt.scatter(qpos_2d[:,0], qpos_2d[:,1], s=5, c='blue', alpha=0.5, label="qpos")
    plt.scatter(ee_2d[:,0], ee_2d[:,1], s=5, c='red', alpha=0.5, label="ee")

    # 连线（连接对应的数据点）
    for i in range(len(qpos)):
        plt.plot([qpos_2d[i,0], ee_2d[i,0]], [qpos_2d[i,1], ee_2d[i,1]], 
                 c='gray', linewidth=0.5, alpha=0.3)

    plt.legend()
    plt.title("Correspondence between qpos and ee")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved correspondence plot to {save_path}")

# def compute_loss():


def main(plot_tsne=False, plot_correspondence=False, compute_loss=True):
    parser = argparse.ArgumentParser(description='single-process evaluation on Calvin bench')
    parser.add_argument("--host", default='0.0.0.0', help="Your client host ip")
    parser.add_argument("--port", default=8000, type=int, help="Your client port")
    # parser.add_argument("--stats_path", default='', type=str, help="Your global stats file for relative data / discrete models")
    args = parser.parse_args()

    name = 'abs_ee'
    data_type = 'abs'
    # Example usage
    stats_file = None
    metas_path = "/home/fyc/EmpiricalStudyForVLA/datasets/meta_files/" + name + "_single_camera-50-10.jsonl"
    save_folder = "distributions"
    stats_file = "/home/fyc/EmpiricalStudyForVLA/datasets/meta_files/" + name + "_single_camera-50-10_global_stats_" + data_type + ".npz"
    print('stats_file', stats_file)
    eef_all = collect_all_actions(metas_path, stats_file, compute_loss=True, args=args)
    print('eef_all[:, :3]', eef_all[:, :3].shape)
    eef_euler = np.concatenate([
        eef_all[:, :3],
        rot6_to_euler(eef_all[:, 3:9]),                        # (T,7)
        eef_all[:, 10][:, None],             # (T,1)
        eef_all[:, 10:13],
        rot6_to_euler(eef_all[:, 13:19]),                       # (T,7)
        eef_all[:, 19][:, None]           # (T,1)
    ], axis=-1)

    name = 'abs_qpos'
    # Example usage
    stats_file = None
    metas_path = "/home/fyc/EmpiricalStudyForVLA/datasets/meta_files/" + name + "_single_camera-50-10.jsonl"
    save_folder = "distributions"
    stats_file = "/home/fyc/EmpiricalStudyForVLA/datasets/meta_files/" + name + "_single_camera-50-10_global_stats_" + data_type + ".npz"
    print('stats_file', stats_file)
    qpos_all = collect_all_actions(metas_path, stats_file, compute_loss=True, args=args)

    qpos_ent, qpos_gau = compute_entropy(qpos_all)
    print('qpos entropy', qpos_ent, qpos_gau)    
    ee_ent0, ee_gau = compute_entropy(eef_all)
    print('ee entropy', ee_ent0, ee_gau)
    ee_ent, ee_gau = compute_entropy(eef_euler)
    print('eef_euler entropy', ee_ent, ee_gau)
    plot_entropy_bar(ee_ent, qpos_ent, data_type, "entropy_bar_" + data_type + ".png")
    plot_entropy_heatmap(ee_ent, qpos_ent, "entropy_heatmap_" + data_type + ".png")

    if plot_correspondence:
        plot_qpos_eef_correspondence(qpos_all, eef_euler, save_path=save_folder + "/qpos_eef_correspondence.png")

    if plot_tsne:
        file_name = metas_path.split('/')[-1].split('.')[0] + '.png'
        print(file_name)
        plot_action_dim_distributions(collected, os.path.join(save_folder, file_name))
        plot_tsne(collected, save_folder)


    # plot_pca(collected, save_path)

if __name__ == "__main__":
    main()