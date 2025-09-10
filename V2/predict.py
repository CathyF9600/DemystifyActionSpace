from mmengine import fileio
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import matplotlib.pyplot as plt
import timm
import cv2
# 🔹 根据你的注册函数加载
from model import model_abs_ee_cnt30   # 或者其他 model_xxx
from safetensors.torch import load_file
from mmengine import fileio
import io

# === 图像预处理 ===
image_aug = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
])
def decode_image_from_bytes(camera_rgb_image):
    """解码单帧字节流为 RGB 图像 (H,W,3)"""
    img = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)  # BGR
    return img


LANG_MAP = {
    "Shake the bottle with proper arm.": "shake bottle",
    "Open the laptop.": "open laptop",
    # 你可以扩充这个映射
}

def load_episode(hdf5_path):
    # with h5py.File(hdf5_path, 'r') as f:
    value = fileio.get(hdf5_path)
    f = io.BytesIO(value)
    h = h5py.File(f,'r')
    raw_images = h['observations/images/cam_high'][()]
    images = [decode_image_from_bytes(x) for x in raw_images]

    proprio = h['observations/eef_6d'][()]
    gt_actions = h['observations/eef_6d'][()]

    lang = h['language_instruction'][()]
    if isinstance(lang, bytes): lang = lang.decode("utf-8")
    if lang in LANG_MAP:
        lang = LANG_MAP[lang]  # 👈 转换成 encoded_language.pt 里的 key
    
    return images, proprio, gt_actions, lang


def predict_episode_chunks(hdf5_path, model, lang_encoder, device="cuda", chunk_size=10, stats=None):
    global_mean = np.asarray(stats["mean"])
    global_std = np.asarray(stats["std"])
    global_min = np.asarray(stats["min"])
    global_max = np.asarray(stats["max"])

    images, proprio, gt_actions, lang = load_episode(hdf5_path)
    
    # gt_actions = (gt_actions - global_mean[None, :])/(global_std[None, :] + 1e-8)

    encoded_lang = torch.tensor(lang_encoder.encode_language(lang),
                                dtype=torch.float32).unsqueeze(0).to(device)

    preds = []
    with torch.no_grad():
        for start in range(0, len(images)-chunk_size, chunk_size):
            # 取 chunk_size 张图像
            imgs = [Image.fromarray(images[start]).convert("RGB")]
            imgs_t = torch.stack([image_aug(img) for img in imgs])  # (chunk,3,H,W)
            imgs_t = imgs_t.unsqueeze(0).to(device)                 # (B=1, V=chunk, C,H,W)

            proprio_t = torch.tensor(proprio[start], dtype=torch.float32).unsqueeze(0).to(device)
            # prorpio_seq = (proprio_t - torch.tensor(global_mean[None, :]).to(device)) / (torch.tensor(global_std[None, :]).to(device) + 1e-8)
            
            pred = model.pred_action(
                images=imgs_t,
                encoded_language=encoded_lang,
                proprio=proprio_t
            )   # (1, chunk_size, dim_actions)
            # print('preds', preds)
            
            preds.append(pred.squeeze(0).cpu().numpy())  # (chunk, dim_actions)

    preds = np.concatenate(preds, axis=0)  # (T', dim_actions)
    print(preds.shape)
    return gt_actions[:preds.shape[0]], preds, lang


import matplotlib.pyplot as plt
import os

def plot_actions(gt_actions, pred_actions, lang, save_dir="./plots", prefix="episode"):
    os.makedirs(save_dir, exist_ok=True)

    T, A = gt_actions.shape
    fig, axes = plt.subplots(A, 1, figsize=(12, 2*A), sharex=True)
    if A == 1: axes = [axes]

    for dim in range(A):
        axes[dim].plot(gt_actions[:, dim], label="GT", color="blue")
        axes[dim].plot(pred_actions[:, dim], label="Pred", color="red", linestyle="--")
        axes[dim].set_ylabel(f"Dim {dim}")
        axes[dim].legend()

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"Prediction vs GT (lang={lang})")
    plt.tight_layout()

    # === 保存图片 ===
    save_path = os.path.join(save_dir, f"{prefix}_prediction.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)   # 关闭避免内存累积
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 🔹 选择你需要的模型
    model, lang_encoder = model_abs_ee_cnt30()
    model = model.to(device)
    model.eval()
    
    ckpt_path='/home/agilex/empirical/abs_ee_aug26_act30/model.safetensors'
    ckpt = load_file(ckpt_path)
    print(model.load_state_dict(ckpt, strict=False))
    model.to(torch.float32).to(device)

    for idx in range(100):
        stats_file = "/home/agilex/empirical/abs_ee_aug26_act30/real_abs_ee_global_stats_abs.npz"
        print('Loading mean and std from', stats_file)
        stats = np.load(stats_file)

        hdf5_path = f"/home/agilex/data_processed/robotwin/shake_bottle_0823/episode_{idx}.hdf5"
        gt, pred, lang = predict_episode_chunks(hdf5_path, model, lang_encoder, device, chunk_size=30, stats=stats)
        #hyper parameter is 0.1,
        
        plot_actions(gt, pred, lang, save_dir="./results", prefix=f"episode_{idx}")


        gt = torch.tensor(gt, dtype=torch.float32, device=device)
        pred = torch.tensor(pred, dtype=torch.float32, device=device)

        delta = 0.1
        loss = F.huber_loss(pred, gt, delta=delta, reduction="mean")

        print("Huber loss:", loss.item())
