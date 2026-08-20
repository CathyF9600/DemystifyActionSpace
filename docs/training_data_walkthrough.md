# Training Data Walkthrough

This guide walks a new user through the two training paths in this repo with real downloaded data:

1. Download one real-world task from `cfeng9600/DemystifyActionSpace` and run the top-level `train.sh`.
2. Download a small RoboTwin sample task and run `robotwin/torchrun.sh`.

The commands below assume the repository is cloned locally and the current directory is the repository root.

```bash
git clone https://github.com/CathyF9600/DemystifyActionSpace.git
cd DemystifyActionSpace
```

## 0. Environment

Use Python 3.10. The package metadata currently declares `python_requires=">=3.10,<3.11"`.

```bash
conda create -n das python=3.10 -y
conda activate das
pip install -e .
pip install "huggingface_hub[cli]"
```

If you are on a machine where `python` does not exist, the shell scripts in this repo also accept an explicit interpreter:

```bash
PYTHON=python3 bash train.sh
```

For large Hugging Face downloads, login is optional for public data but recommended to avoid rate limits:

```bash
hf auth login
```

## 1. Real-World Touch Task -> `train.sh`

The real-world dataset lives at:

- Dataset page: https://huggingface.co/datasets/cfeng9600/DemystifyActionSpace
- Touch task browser: https://huggingface.co/datasets/cfeng9600/DemystifyActionSpace/tree/main/touch

The `touch/` directory contains HDF5 trajectories such as:

```text
touch/touch_cube/episode_0.hdf5
touch/touch_cube/episode_1.hdf5
...
```

### 1.1 Download one task

Download only the `touch` task into a local data directory:

```bash
mkdir -p data/hf
hf download cfeng9600/DemystifyActionSpace \
  --repo-type dataset \
  --include "touch/**" \
  --local-dir data/hf/DemystifyActionSpace
```

For a quicker first run, download only one subfolder:

```bash
mkdir -p data/hf
hf download cfeng9600/DemystifyActionSpace \
  --repo-type dataset \
  --include "touch/touch_cube/**" \
  --local-dir data/hf/DemystifyActionSpace
```

### 1.2 Build the meta file expected by `train.sh`

The top-level loader expects `--train_metas_path` to point to a JSON file with:

```json
{
  "task_name": "touch cube",
  "datalist": ["/absolute/path/to/episode_0.hdf5"]
}
```

`train.sh` already points at `real-world/meta_files/touch.jsonl`, so create that file from the downloaded HDF5s:

```bash
mkdir -p real-world/meta_files
python - <<'PY'
from pathlib import Path
import json

repo = Path.cwd()
data_root = repo / "data/hf/DemystifyActionSpace/touch"
paths = sorted(data_root.rglob("*.hdf5"))
if not paths:
    raise SystemExit(f"No HDF5 files found under {data_root}")

meta = {
    "task_name": "touch cube",
    "datalist": [str(p.resolve()) for p in paths],
}

out = repo / "real-world/meta_files/touch.jsonl"
out.write_text(json.dumps(meta, indent=2))
print(f"Wrote {len(paths)} trajectories to {out}")
PY
```

The checked-in `assets/encoded_language.pt` contains the `touch cube` key used by this meta file.

### 1.3 Run top-level training

For the full default run:

```bash
bash train.sh
```

For a short first pass, override the same arguments manually:

```bash
PYTHON=${PYTHON:-python3}
$PYTHON train.py \
  --model model_abs_ee_act \
  --epochs 1 \
  --sample_num 32 \
  --batch-size 4 \
  --precision no \
  --learning_rate 3e-4 \
  --output_dir real-world/exp/touch_cube/abs_ee_debug \
  --train_metas_path real-world/meta_files/touch.jsonl \
  --save_interval 1000 \
  --port 29550 \
  --weight_decay 0
```

Expected progress:

```text
successfully load language hub: dict_keys([...])
number of params: ... M
================detect dataset with traj ...==================
Start training for ... ep
[epoch] 0 [Iter 0] [Training Loss] ...
```

Checkpoints are written under the `--output_dir`, for example:

```text
real-world/exp/touch_cube/abs_ee_debug/ckpt-final/
```

## 2. RoboTwin Sample Data -> `robotwin/torchrun.sh`

This repo's RoboTwin loader expects raw RoboTwin 2.0 HDF5 files with keys including:

```text
endpose/left_endpose
endpose/right_endpose
endpose/left_gripper
endpose/right_gripper
joint_action/left_arm
joint_action/right_arm
joint_action/left_gripper
joint_action/right_gripper
observation/<camera>/rgb
```

Use the official RoboTwin 2.0 Hugging Face dataset:

- Dataset page: https://huggingface.co/datasets/TianxingChen/RoboTwin2.0
- Task browser: https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset/click_bell

That dataset provides task archives such as:

```text
dataset/click_bell/aloha-agilex_clean_50.zip
```

Each archive contains RoboTwin 2.0 HDF5 trajectories and instruction files for the selected task.

### 2.1 Download and extract one sample task

`dataset/click_bell/aloha-agilex_clean_50.zip` is a small clean RoboTwin 2.0 task archive.

```bash
mkdir -p data/robotwin_downloads data/robotwin_raw
hf download TianxingChen/RoboTwin2.0 \
  --repo-type dataset \
  --include "dataset/click_bell/aloha-agilex_clean_50.zip" \
  --local-dir data/robotwin_downloads

unzip -q data/robotwin_downloads/dataset/click_bell/aloha-agilex_clean_50.zip \
  -d data/robotwin_raw/click_bell
```

After extraction, find the RoboTwin 2.0 HDF5 files:

```bash
find data/robotwin_raw -name 'episode*.hdf5' | head
```

You should see `episode*.hdf5` files. The exact parent directory can vary by archive, so the later meta-generation step searches recursively.

### 2.2 Create a training view with the expected task name

`robotwin/dataset.py` derives the language key from the third parent directory of each HDF5 path:

```python
ins = datapath.split('/')[-3].replace('_', ' ')
```

For that reason, create a lightweight symlink view whose path shape is:

```text
data/robotwin_train_view/click_bell/episodes/episode_000000.hdf5
```

Then `-3` is `click_bell`, and the language key becomes `click bell`.

```bash
mkdir -p data/robotwin_train_view/click_bell/episodes
while IFS= read -r -d '' episode; do
  ln -sf "$(realpath "$episode")" data/robotwin_train_view/click_bell/episodes/
done < <(find data/robotwin_raw -name 'episode*.hdf5' -print0)
```

Confirm the view:

```bash
find data/robotwin_train_view/click_bell/episodes -name 'episode*.hdf5' | head
```

### 2.3 Build a RoboTwin meta file

The RoboTwin loader expects a JSON meta file with `dataset_name`, `datalist`, and `observation_key`.

Use a single camera for `rel_ee_cnt_rot`, because that registered model is configured with `num_views=1`.

```bash
mkdir -p robotwin/meta_files
python - <<'PY'
from pathlib import Path
import json

repo = Path.cwd()
paths = sorted((repo / "data/robotwin_train_view/click_bell/episodes").glob("episode*.hdf5"))
if not paths:
    raise SystemExit("No RoboTwin HDF5 files found. Check the extraction and symlink steps.")

meta = {
    "dataset_name": "robotwin2_rel_ee",
    "datalist": [str(p.resolve()) for p in paths],
    "observation_key": ["observation/head_camera/rgb"],
}

out = repo / "robotwin/meta_files/click_bell_rel_ee_single_camera.jsonl"
out.write_text(json.dumps(meta, indent=2))
print(f"Wrote {len(paths)} trajectories to {out}")
PY
```

If your sample uses a different camera name, inspect one episode:

```bash
python - <<'PY'
from pathlib import Path
import h5py
p = next(Path("data/robotwin_train_view/click_bell/episodes").glob("episode*.hdf5"))
with h5py.File(p, "r") as f:
    print("observation groups:", list(f["observation"].keys()))
PY
```

Then update `observation_key` to match, for example `observation/front_camera/rgb`.

### 2.4 Build language embeddings for the sample task

`robotwin/train.py` takes `--pt_path`, a PyTorch file mapping language strings to 768-dimensional embeddings.
For this walkthrough the derived language key is `click bell`.

```bash
python - <<'PY'
from pathlib import Path
import torch
from transformers import AutoTokenizer, SiglipTextModel

text = "click bell"
model_id = "google/siglip-base-patch16-224"

tokenizer = AutoTokenizer.from_pretrained(model_id)
text_model = SiglipTextModel.from_pretrained(model_id)
text_model.eval()

with torch.no_grad():
    tokens = tokenizer([text], padding=True, return_tensors="pt")
    emb = text_model(**tokens).pooler_output[0].cpu()

out = Path("robotwin/encoded_language_click_bell.pt")
torch.save({text: emb}, out)
print(f"Wrote {out} with key {text!r} and shape {tuple(emb.shape)}")
PY
```

Expected shape:

```text
shape (768,)
```

If you train a different RoboTwin task, make sure the key in this file exactly matches the key derived from the HDF5 path.
For example, a path containing `handover_block` as the third parent produces `handover block`.

### 2.5 Run RoboTwin training

`robotwin/torchrun.sh` can be configured with environment variables. Start with one GPU/process until the data path is known-good:

```bash
WANDB_MODE=disabled \
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
train_metas_path="$PWD/robotwin/meta_files/click_bell_rel_ee_single_camera.jsonl" \
pt_path="$PWD/robotwin/encoded_language_click_bell.pt" \
output_dir="$PWD/runnings/robotwin/click_bell_rel_ee_cnt_rot" \
iters=100 \
batch_size=4 \
bash robotwin/torchrun.sh
```

Expected progress:

```text
model init
rank 0
================detect dataset robotwin2_rel_ee with traj ...==================
len(hdf5_files) ...
successfully load language hub from ... encoded_language_click_bell.pt: dict_keys(['click bell'])
>>>>>>>> compute_mean_std
Start training: global iter 0 .. 99
[Iter 0] [Training Loss] ...
```

For the longer default run, increase `iters`, `batch_size`, and `NPROC_PER_NODE` to match your GPU count:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
train_metas_path="$PWD/robotwin/meta_files/click_bell_rel_ee_single_camera.jsonl" \
pt_path="$PWD/robotwin/encoded_language_click_bell.pt" \
output_dir="$PWD/runnings/robotwin/click_bell_rel_ee_cnt_rot" \
iters=80000 \
batch_size=32 \
bash robotwin/torchrun.sh
```

## 3. Common Checks

### HDF5 field check

Use this if a loader fails with `KeyError`:

```bash
python - <<'PY'
from pathlib import Path
import h5py

p = next(Path("data/robotwin_train_view/click_bell/episodes").glob("episode*.hdf5"))
with h5py.File(p, "r") as f:
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name, obj.shape, obj.dtype)
    f.visititems(visit)
PY
```

### Language key check

Use this if training fails with `KeyError` from `self.language_emb[ins]`:

```bash
python - <<'PY'
from pathlib import Path
import torch

pt = torch.load("robotwin/encoded_language_click_bell.pt", map_location="cpu")
print("available language keys:", list(pt.keys()))
for p in Path("data/robotwin_train_view").glob("*/*/*.hdf5"):
    print(p, "->", str(p).split("/")[-3].replace("_", " "))
    break
PY
```

The printed derived key must be present in the `.pt` file.

### Multi-camera note

`rel_ee_cnt_rot` is configured for one view. If you set:

```json
"observation_key": ["observation/head_camera/rgb", "observation/left_camera/rgb"]
```

you must also use or register a model variant with `num_views=2`; otherwise the decoder input dimension will not match.

## 4. What Each Script Uses

Top-level `train.sh` defaults:

```text
--model model_abs_ee_act
--train_metas_path real-world/meta_files/touch.jsonl
--output_dir real-world/exp/touch_cube/abs_ee
```

RoboTwin `robotwin/torchrun.sh` defaults can be overridden with:

```text
name
port
output_dir
train_metas_path
pt_path
iters
save_interval
batch_size
precision
learning_rate
weight_decay
NPROC_PER_NODE
PYTHON
```
