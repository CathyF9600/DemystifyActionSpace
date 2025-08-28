# put this under RoboTwin/script/robotwin_client_v2.py
import sys
START = 0
sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs.utils.create_actor import UnStableError
# Add the root directory (RoboTwin/) to sys.path
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent  # Adjust based on actual structure
sys.path.append(str(root_dir))
# print(f"sys.path: {sys.path}")
import argparse
import collections
from collections import Counter, defaultdict
import logging
import os
import importlib
import numpy as np
import torch
import yaml
import json_numpy
import requests
import PIL.Image as Image 
from tqdm import tqdm
import traceback
import json
import random
import imageio
import cv2
import sys
import numpy as np
from generate_episode_instructions import *
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# os.chdir(os.path.join(os.getcwd(), 'eval/robotwin/RoboTwin'))

DATA_TYPE, ACTION_TYPE = None, None
logger = logging.getLogger(__name__)

ALL_TASKS = [
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller", "handover_block",
    "handover_mic", "hanging_mug", "lift_pot", "move_can_pot", "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop", "open_microwave",
    "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left", "place_a2b_right",
    "place_bread_basket", "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate", "place_dual_shoes", "place_empty_cup",
    "place_fan", "place_mouse_pad", "place_object_basket", "place_object_scale",
    "place_object_stand", "place_phone_stand", "place_shoe", "press_stapler",
    "put_bottles_dustbin", "put_object_cabinet", "rotate_qrcode", "scan_object",
    "shake_bottle_horizontally", "shake_bottle", "stack_blocks_three", "stack_blocks_two",
    "stack_bowls_three", "stack_bowls_two", "stamp_seal", "turn_switch"
]

# TEST_TASKS = ["adjust_bottle"]

def quat_to_rotate6D(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def rotate6D_to_quat(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_quat()



def decode_image_from_bytes(camera_rgb_image):
    if isinstance(camera_rgb_image, (bytes, bytearray)):
        camera_rgb_image = np.frombuffer(camera_rgb_image, np.uint8)
    return cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)

def save_results(path, task_name, rewards, video_records):
    video_path = os.path.join(path, f"{task_name}")
    json_path = os.path.join(path, 'results.json')
    # save metrics
    os.makedirs(path, exist_ok=True)
    metrics = {f"sim/{task_name}": rewards}
    with open(json_path, 'a+') as f:
        line = json.dumps(metrics)
        f.write(line+'\n')
    # save videos
    os.makedirs(video_path, exist_ok=True)
    for k,v in video_records.items():
        B, H, W, C = v.shape
        video_save_path = os.path.join(video_path, f"{k}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter(video_save_path, fourcc, 25, (W, H))
        for frame in v:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()

class ClientModel:
    def __init__(self, host, port):
        self.url = f"http://{host}:{port}/act"
        self.vision_record = []
        self.proprio = None
        
    def set_instruction(self, instruction):
        self.instruction = instruction
        
    def return_vision_record(self):
        video = np.stack(self.vision_record)
        self.vision_record = []
        return video
    
    def step(self, obs):
        global TYPE
        head_view = obs['observation']['head_camera']['rgb']
        left_view = obs['observation']['left_camera']['rgb']
        right_view = obs['observation']['right_camera']['rgb']
        front_view = obs['observation']['front_camera']['rgb']
        # image_obs = np.stack([head_view, left_view, right_view, front_view])[None, ]
        if self.proprio is None:
            if ACTION_TYPE == 'ee':
                left_ee = np.expand_dims(np.array(obs["endpose"]["left_endpose"]), axis=0)     # shape (T, 7)
                right_ee = np.expand_dims(np.array(obs["endpose"]["right_endpose"]), axis=0)    # shape (T, 7)
                left_grip = np.expand_dims(np.array(obs["endpose"]["left_gripper"]), axis=0)    # shape (T,)
                right_grip = np.expand_dims(np.array(obs["endpose"]["right_gripper"]), axis=0)  # shape (T,)
                # left_grip = 1 - left_grip * 2
                # right_grip = 1 - right_grip * 2
                self.proprio = np.concatenate([
                    left_ee[:, :3],
                    quat_to_rotate6D(left_ee[:, 3:]),                        # (T,7)
                    left_grip[:, None],             # (T,1)
                    right_ee[:, :3],
                    quat_to_rotate6D(right_ee[:, 3:]),                       # (T,7)
                    right_grip[:, None]             # (T,1)
                ], axis=-1).squeeze(0)
            # self.vision_record.append(image_obs)
            else: # qpos
                print('Getting qpos data', obs['joint_action'])
                left_arm = np.expand_dims(np.array(obs["joint_action"]["left_arm"]), axis=0)     # shape (T, 7)
                right_arm = np.expand_dims(np.array(obs["joint_action"]["right_arm"]), axis=0)     # shape (T, 7)
                left_grip = np.expand_dims(np.array(obs["joint_action"]["left_gripper"]), axis=0)
                right_grip = np.expand_dims(np.array(obs["joint_action"]["right_gripper"]), axis=0)
                # left_grip = 1 - left_grip * 2
                # right_grip = 1 - right_grip * 2
                proprio0 = obs['joint_action']['vector'][None]
                self.proprio = np.concatenate([
                    left_arm,
                    left_grip[:, None],             # (T,1)
                    right_arm,
                    right_grip[:, None]             # (T,1)
                ], axis=-1).squeeze(0)
                print('shape checking', proprio.shape, proprio0.shape)
        query = {
                # "proprio": json_numpy.dumps(np.zeros_like(self.proprio)), #.reshape(1,-1),  # (1, 14)
                "proprio": json_numpy.dumps(self.proprio),
                "language_instruction": self.instruction,
                "image0": json_numpy.dumps(head_view),
                "image1": json_numpy.dumps(left_view),
                "image2": json_numpy.dumps(right_view)}
        # print('ACTION_TYPE', ACTION_TYPE, 'DATA_TYPE', DATA_TYPE)
        if DATA_TYPE == 'rel':
            query['data_type'] = 'rel'
        else:
            query['data_type'] = 'abs'
        if ACTION_TYPE == 'ee':
            query['action_type'] = 'ee'
        else:
            query['action_type'] = 'rel'
        response = requests.post(self.url, json=query).json()
        print('response.json()', response.keys())
        if DATA_TYPE == 'abs':
            noise_action = np.array(response['action']).squeeze(0)
            self.proprio = noise_action[-1]
            # left_action = np.array(response['left_action'])
            # right_action = np.array(response['right_action'])
            print('noise_action', noise_action)
        # action[:, -1] = np.where(action[:, -1] > 0.02, 0.045, -0.01)
        # action[:, 6] = np.where(action[:, 6] > 0.02, 0.045, -0.01)
        else:
            noise_action = np.array(response['action_sum'])
            # print('noise_action', noise_action.shape)
            self.proprio = noise_action[-1]
            # left_action = np.array(response['left_action'])
            # right_action = np.array(response['right_action'])
            print('current action', noise_action)
        return noise_action

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_env(task_name, task_config):
    CONFIGS_PATH = "task_config"
    task = class_decorator(task_name)

    config_path = os.path.join(CONFIGS_PATH, f"{task_config}.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("missing embodiment files")
        return robot_file

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    args["embodiment_name"] = "+".join(embodiment_type) if len(embodiment_type) > 1 else embodiment_type[0]
    args["task_config"] = task_config

    return task, args

def init_env(task_name):
    """_summary_
    initialize your env given a specific task name
    """
    with open(f'./conf/task_config/{task_name}.yml', 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    head_camera_config = get_camera_config(args['head_camera_type'])
    args['head_camera_fovy'] = head_camera_config['fovy']
    args['head_camera_w'] = head_camera_config['w']
    args['head_camera_h'] = head_camera_config['h']
    head_camera_config = 'fovy' + str(args['head_camera_fovy']) + '_w' + str(args['head_camera_w']) + '_h' + str(args['head_camera_h'])
    
    wrist_camera_config = get_camera_config(args['wrist_camera_type'])
    args['wrist_camera_fovy'] = wrist_camera_config['fovy']
    args['wrist_camera_w'] = wrist_camera_config['w']
    args['wrist_camera_h'] = wrist_camera_config['h']
    wrist_camera_config = 'fovy' + str(args['wrist_camera_fovy']) + '_w' + str(args['wrist_camera_w']) + '_h' + str(args['wrist_camera_h'])

    front_camera_config = get_camera_config(args['front_camera_type'])
    args['front_camera_fovy'] = front_camera_config['fovy']
    args['front_camera_w'] = front_camera_config['w']
    args['front_camera_h'] = front_camera_config['h']
    front_camera_config = 'fovy' + str(args['front_camera_fovy']) + '_w' + str(args['front_camera_w']) + '_h' + str(args['front_camera_h'])

    # output camera config
    print('============= Camera Config =============')
    print('Head Camera Config:\n    type: '+ str(args['head_camera_type']) + '       fovy: ' + str(args['head_camera_fovy']) + '\n    camera_w: ' + str(args['head_camera_w']) + '    camera_h: ' + str(args['head_camera_h']))
    print('Wrist Camera Config:\n    type: '+ str(args['wrist_camera_type']) + '       fovy: ' + str(args['wrist_camera_fovy']) + '\n    camera_w: ' + str(args['wrist_camera_w']) + '    camera_h: ' + str(args['wrist_camera_h']))
    print('Front Camera Config:\n    type: '+ str(args['front_camera_type']) + '       fovy: ' + str(args['front_camera_fovy']) + '\n    camera_w: ' + str(args['front_camera_w']) + '    camera_h: ' + str(args['front_camera_h']))
    print('=======================================')
    
    env = class_decorator(args['task_name'])

    return env, args

def build_instruction_lib(task_name):
    ins_lib_path = f"./conf/instructions/{task_name}.json"
    assert os.path.isfile(ins_lib_path), f"Instruction file of {task_name} is missing"
    ins_lib = json.load(open(ins_lib_path, 'r'))["instructions"]
    return ins_lib


def interpolate_gripper(gripper, new_steps):
    n = gripper.shape[0]  # original step
    x_old = np.linspace(0, 1, n)  # original data
    x_new = np.linspace(0, 1, new_steps)  # target data number
    
    # use numpy.interp to linear interpolate
    gripper_interp = np.interp(x_new, x_old, gripper.flatten()).reshape(-1)
    return gripper_interp

#### debug functions ####
# 设置打印选项（全局生效）
np.set_printoptions(
    precision=4,      # 保留4位小数
    suppress=True,    # 禁用科学计数法
    linewidth=120     # 每行最大宽度
)

def print_action_human_readable(action, prefix="action"):
    """格式化打印动作数据，区分左右臂并添加标签"""
    # 假设动作格式: [left_xyz(3), left_quat(4), left_gripper(1), 
    #               right_xyz(3), right_quat(4), right_gripper(1)]
    left = action[:8]
    right = action[8:]
    
    # 提取各部分
    left_xyz = left[:3]
    left_quat = left[3:7]
    left_gripper = left[7]
    
    right_xyz = right[:3]
    right_quat = right[3:7]
    right_gripper = right[7]
    
    # 打印格式化数据
    # print(f"\n{prefix}:")
    # print(f"  Left Arm:")
    # print(f"    Position:  [{left_xyz[0]:.4f}, {left_xyz[1]:.4f}, {left_xyz[2]:.4f}]")
    # print(f"    Orientation: [{left_quat[0]:.4f}, {left_quat[1]:.4f}, {left_quat[2]:.4f}, {left_quat[3]:.4f}]")
    # print(f"    Gripper: {left_gripper:.4f}")
    
    # print(f"  Right Arm:")
    # print(f"    Position:  [{right_xyz[0]:.4f}, {right_xyz[1]:.4f}, {right_xyz[2]:.4f}]")
    # print(f"    Orientation: [{right_quat[0]:.4f}, {right_quat[1]:.4f}, {right_quat[2]:.4f}, {right_quat[3]:.4f}]")
    # print(f"    Gripper: {right_gripper:.4f}")

#########################

def _rollout(env, policy):
    # obs = env.reset()
    success_flag = False
    error_flag = False
    env._update_render()
    # print('env.render_freq', env.render_freq)
    if env.render_freq: env.viewer.render()
    env.actor_pose = True

    images = []
    idx = 0
    # obs = env.get_obs() # get observation
    policy.proprio = None
    #current_state = None
    for j in range(30): #env.step_lim: # If it is not successful within the specified number of steps, it is judged as a failure.
        obs = env.get_obs() # get observation
        actions = policy.step(obs)
        if ACTION_TYPE == 'ee':
            left_xyz = actions[:, :3]  # (B, 3)
            left_rotate6d = actions[:, 3:9]  # (B, 6)
            left_gripper = actions[:, 9:10]  # (B, 1)  # Use 9:10 to keep 2D shape
            # left_gripper = np.where(left_gripper > 0.5, 1.0, 0.0)

            # Convert 6D rotation to quaternion (4 elements)
            left_quat = rotate6D_to_quat(left_rotate6d)  # Should return (B, 4)
            # left_gripper = (1 - left_gripper ) / 2
            # Ensure all have shape (B, features) and concatenate along axis 1
            left_new = np.concatenate([left_xyz, left_quat, left_gripper], axis=1)  # (B, 3+4+1) = (B, 8)

            # Do the same for right arm
            right_xyz = actions[:, 10:13]  # (B, 3)
            right_rotate6d = actions[:, 13:19]  # (B, 6)
            right_quat = rotate6D_to_quat(right_rotate6d)  # (B, 4)
            right_gripper = actions[:, 19:20]  # (B, 1)
            # right_gripper = np.where(right_gripper > 0.5, 1.0, 0.0)

            # 还原夹爪值的原始状态
            # right_gripper = (1 - right_gripper) / 2
            right_new = np.concatenate([right_xyz, right_quat, right_gripper], axis=1)  # (B, 8)
            # Final rollout action (16 dimensions total)
            rollout_action = np.concatenate([left_new, right_new], axis=1)  # (B, 16)
        else: # qpos
            # print('******actions.shape', actions.shape)

            # actions[6] = (1 - actions[6]) /2
            # actions[-1] = (1- actions[-1]) /2 
            rollout_action = actions
        for action in tqdm(rollout_action):
            # with open("/home/dodo/fyc/RoboTwin/log/log.txt", "a+") as f:
            #     f.write(f"{j} {idx} current action {action} \n")
            # print('ACTION_TYPE', ACTION_TYPE)
            if ACTION_TYPE == 'ee':
                env.take_action(action, action_type='ee')  # target_pose: np.array([x, y, z, qx, qy, qz, qw])
            else:
                env.take_action(action, action_type='qpos') 
            obs = env.get_obs()
            # obs['endpose']['left_endpose'] = list(left_new)
            # obs['endpose']['right_endpose'] = list(right_new)
            images.append(obs["observation"]["head_camera"]["rgb"] )
            idx += 1
            if env.check_success():
                # print("\nsuccess!")
                # with open("/home/dodo/fyc/RoboTwin/log/log.txt", "a+") as f:
                #     f.write(f"\nsuccess!\n")
                # env.suc +=1
                # return 1, images

                success_flag = True
                break
            if env.actor_pose == False: 
                print('false actor_pose')
                error_flag = True
                break
        # # print(f'step: {step_cnt} / {env.step_lim}', end='\r')
        # if error_flag:
        #     print("\nfail due to false actor_pose!")
        #     return 0, images

        if success_flag:
            print("\nsuccess!")
            env.suc +=1
            return 1, images
        
        if env.actor_pose == False:
            print('false actor_pose2')
            break
        # current_state = obs
        j += 1
        env._update_render()
    print("\nfail!")
    # with open("/home/dodo/fyc/RoboTwin/log/log.txt", "a+") as f:
    #     f.write(f"\fail!\n")
    return 0, images

def eval_episodes(task_name, task_config, policy, test_num=10, seed=0, eval_log_dir=None, instruction_type=None):
    """
    rollout several episodes and log the mean episode return
    """
    if not os.path.exists(os.path.join(eval_log_dir, task_name)):
        print('save to', os.path.join(eval_log_dir, task_name))
        os.makedirs(os.path.join(eval_log_dir, task_name))
    
    # env, args = init_env(task_name)
    TASK_ENV, args = load_env(task_name, task_config)

    # ins_lib = build_instruction_lib(task_name)
    st_seed = 100000 * (1 + seed)
    suc_nums = []
    topk = 1
    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]
    args['policy_name'] = 'V4'

    args["eval_mode"] = True
    args["render_freq"] = 0
    args['ckpt_setting'] = '60k'
    while succ_seed < test_num:
        render_freq = args["render_freq"]
        print('Running test', now_id)
        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except Exception as e: # UnStableError as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        # results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = args["task_name"].replace('_', ' ') # np.random.choice(results[0][instruction_type])
        print('instruction:', instruction)
        policy.set_instruction(instruction=instruction)  # set language instruction
        
        status, images = _rollout(TASK_ENV, policy)
        
        # save rollout video
        # if now_id % 10 == 0:
        save_path = f'{eval_log_dir}/{task_name}/{now_id}_{status}.mp4'
        save_video(save_path, images)
        # log per episode results
        metrics = {f'sim/{task_name}': status}
        save_path = f'{eval_log_dir}/results.json'
        _log_results(metrics, save_path)

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        # print('TASK_ENV.render_freq', TASK_ENV.render_freq)
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        # print(
        #     f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
        #     f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        # )
        print(f"Success rate: {round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%")
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc

def save_video(output_path, frames, fps=30):
    print('saving video to ', output_path)
    imageio.mimsave(output_path, frames, fps=fps) 


def _log_results(metrics, log_path):
    # print(metrics)
    with open(log_path, 'a+') as f:
        line = json.dumps(metrics)
        f.write(line+'\n')


def main():
    global DATA_TYPE, ACTION_TYPE
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--host", default='0.0.0.0', help="Your client host ip")
    parser.add_argument("--port", default='8001', help="Your client port")
    parser.add_argument("--eval_log_dir", default='/home/dodo/fyc/HeteroDiffusionPolicy/AbsEEFFlowV4/runnings/RoboTwin/', type=str, help="Where to log the evaluation results.")
    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    parser.add_argument("--num_episodes", default=1000, type=int)
    parser.add_argument("--seed", default=0, type=int)

    # args = parser.parse_args()
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task (envs.<task>)")
    parser.add_argument("--data_type", type=str, required=False, help="abs or rel")
    parser.add_argument("--action_type", type=str, required=True, help="ee or qpos")
    parser.add_argument("--task_config", type=str, required=True, help="Task config name (without .yml)")
    # parser.add_argument("--hdf5_path", type=str, required=True, help="Path to replay HDF5 file")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the output video")
    parser.add_argument("--instruction_type", type=str, required=True, help="Where to save the output video")
    args = parser.parse_args()
    kwargs = vars(args)
    
    # env = load_env(args.task_name, args.task_config)
    # replay_actions_from_hdf5(env, args.hdf5_path, args.output_path)

    DATA_TYPE = kwargs['data_type']
    ACTION_TYPE = kwargs['action_type']
    model = ClientModel(host=kwargs['host'], port=kwargs['port'])
    if args.task_name == 'all':
        for task in tqdm(ALL_TASKS[START:]):
            print(f"Evaluating task {task} for {kwargs['num_episodes']} episodes...")
            rewards = eval_episodes(task_name=task, 
                                    task_config=args.task_config,
                                    policy=model, 
                                    seed=args.seed,
                                    test_num=kwargs['num_episodes'],
                                    eval_log_dir=kwargs['eval_log_dir'],
                                    instruction_type=args.instruction_type)
    else:
        print(f"Evaluating task {args.task_name} for {kwargs['num_episodes']} episodes...")
        rewards = eval_episodes(task_name=args.task_name, 
                                task_config=args.task_config,
                                policy=model, 
                                seed=args.seed,
                                test_num=kwargs['num_episodes'],
                                eval_log_dir=kwargs['eval_log_dir'],
                                instruction_type=args.instruction_type)
if __name__ == "__main__":
    main()