#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import torch
import numpy as np
import os
import time
import pickle
import argparse
from einops import rearrange

# from utils import compute_dict_mean, set_seed, detach_dict # helper functions
# from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
import collections
from collections import deque

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist, Pose, PoseStamped
from sensor_msgs.msg import JointState, Image
from piper_msgs.msg import PosCmd
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import time
import threading
import math
import threading

import json_numpy
import requests

import signal
import cv2
import os
from datetime import datetime
import mediapy

import sys
import threading
import termios
import tty
import select
import h5py
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# from scipy.spatial.transform import Rotation as R
import sys
sys.path.append("./")

from deploy.utils.rotation import eef_6d, eef_quat, abs_6d_2_abs_euler
from deploy.utils.rosoperator import RosOperator

task_config = {'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']}
# TASK = "touch cube"
# TASK = "pick up the red cup"
# TASK  = "pick up the red cup and place it on the plate"
TASK = "Pick up the cube with left arm, pick up the bowl with right arm, and place the cube into the bowl"
SLEEP = False
PORT = '8002' # 2 3
DATA_ORDER = 'rel'
SAMPLE = 60
CHUNK = 60
init_state_id = 0

# data = h5py.File(f"/home/agilex/data_processed/robotwin_new/0916-touch-cube/touch_cube_batch2/episode_{init_state_id}.hdf5",'r')['observations']
# data = h5py.File(f"/home/agilex/data_processed/robotwin_new/pick_cup_all/pick_cup/episode_{init_state_id}.hdf5",'r')['observations']
# data = h5py.File(f"/home/agilex/data_processed/robotwin_new/pick_place_all/pick_place/episode_{init_state_id}.hdf5",'r')['observations']
data = h5py.File(f"/home/agilex/data_processed/robotwin_new/aim_bowl_0921_1/episode_{init_state_id}.hdf5",'r')['observations']
init_qpos = data['qpos'][0].tolist()

replay_data = data['qpos'][:]
print(init_qpos)

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None

current_language_instruction = None
current_data_type = None
current_action_type = None


from scipy.interpolate import interp1d

def upsample_action(action, new_len):
    """
    Upsample an action sequence along the time axis.

    Args:
        action: np.ndarray of shape (T, d_action)
        factor: int, how many times to upsample (default=2 doubles length)

    Returns:
        np.ndarray of shape (T * factor, d_action)
    """
    # task2
    # action[:, 6] = np.where(action[:, 6] > 0.02, 0.04, action[:, 6])
    # action[:, 6] = np.where(action[:, 6] <= 0.01, -0.002, action[:, 6])

    # Task4
    action[:, 6] = np.where(action[:, 6] < 0.03, -0.002, action[:, 6])
    action[:, 13] = np.where(action[:, 13] <= 0.01, -0.002, action[:, 13])
    
    if new_len == action.shape[0]:
        return action
    T, d_action = action.shape
    old_idx = np.linspace(0, 1, T)
    new_idx = np.linspace(0, 1, new_len)

    f = interp1d(old_idx, action, axis=0, kind="linear")
    action_upsampled = f(new_idx)
    return action_upsampled

class ClientModel():
    def __init__(self,
                 host,
                 port):

        self.url = f"http://{host}:{port}/act"
        self.pred_proprio = None
        self.reset()

    def reset(self):
        """
        This is called
        """
        # currently, we dont use historical observation, so we dont need this fc
        
        self.action_plan = collections.deque()
        return None

    def set_proprio(self, proprio):
        # if self.pred_proprio is None:
            self.pred_proprio = proprio

    def step(self, obs, args):
        """
        Args:
            obs: (dict) environment observations
        Returns:
            action: (np.array) predicted action
        """
        if not self.action_plan:
            main_view = obs['images']['cam_high']   #  np.ndarray with shape (480, 640, 3)
            left_wrist_view = obs['images']['cam_left_wrist']   # np.ndarray with shape (480, 640, 3) 
            right_wrist_view = obs['images']['cam_right_wrist']   # np.ndarray with shape (480, 640, 3) 
            proprio = self.pred_proprio.astype(np.float32)
            
            query = {"proprio": json_numpy.dumps(proprio),  # (14,) or (16, )
                    "image0": json_numpy.dumps(main_view),
                    "image1": json_numpy.dumps(left_wrist_view),
                    "image2": json_numpy.dumps(right_wrist_view),
                    'language_instruction': TASK,
                    'data_type': DATA_ORDER,
                    'action_type': 'qpos'
            }
            global current_language_instruction, current_data_type, current_action_type
            current_language_instruction = query['language_instruction']
            current_data_type = query['data_type']
            current_action_type = query['action_type']

            response = requests.post(self.url, json=query)
            action = response.json()['action'][0]
            action = upsample_action(np.array(action), new_len=SAMPLE)

            # print("1111"*88, np.array(action).shape)
            
            # print("action", action)
            self.action_plan.extend(action[:CHUNK])
            self.pred_proprio = np.asarray(self.action_plan[-1]).astype(np.float32)
            print(self.pred_proprio.shape)
        print(len(self.action_plan))
                
        # binary gripper
        action_predict = np.array(self.action_plan.popleft())
        # action_predict[-1] = 0.06894999742507935 if action_predict[-1] > 0.04629657045006752 else 0.001820000004954636
        return action_predict

def get_action(args, config, ros_operator, policy, t, pre_action):
    print_flag = True

    rate = rospy.Rate(args.publish_rate)
    while True and not rospy.is_shutdown():
        # skip the ros query if the action plan is not empty
        if len(policy.action_plan) > 0:
            # print("using exiting chunk with left chunk len:", len(policy.action_plan))
            start_time = time.time()
            all_actions = policy.step(None, args)
            # all_actions = abs_6d_2_abs_euler(all_actions)
        
            end_time = time.time()
            # print("model cost time:", end_time - start_time)
            inference_lock.acquire()
            inference_actions = all_actions
            if pre_action is None:
                pre_action = obs['eef_quat']

            inference_timestep = t
            inference_lock.release()
            return inference_actions #, _, _
            
        # query the camera view if the action plan is empty
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, puppet_arm_left_pose, puppet_arm_right_pose, robot_base, status) = result
        obs = collections.OrderedDict()
        image_dict = dict()

        image_dict[config['camera_names'][0]] = img_front
        image_dict[config['camera_names'][1]] = img_left
        image_dict[config['camera_names'][2]] = img_right


        obs['images'] = image_dict

        if args.use_depth_image:
            image_depth_dict = dict()
            image_depth_dict[config['camera_names'][0]] = img_front_depth
            image_depth_dict[config['camera_names'][1]] = img_left_depth
            image_depth_dict[config['camera_names'][2]] = img_right_depth
            obs['images_depth'] = image_depth_dict
            
        obs['eef_quat'] = eef_quat(puppet_arm_left, puppet_arm_right, puppet_arm_left_pose, puppet_arm_right_pose)
        obs['eef_6d'] = eef_6d(puppet_arm_left, puppet_arm_right, puppet_arm_left_pose, puppet_arm_right_pose)
        obs['qpos'] = np.concatenate(
            (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
        obs['qvel'] = np.concatenate(
            (np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
        obs['effort'] = np.concatenate(
            (np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
        if args.use_robot_base:
            obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
        else:
            obs['base_vel'] = [0.0, 0.0]

        start_time = time.time()
        # policy.set_proprio(obs['eef_6d'])
        policy.set_proprio(obs['qpos'])
        all_actions = policy.step(obs, args)
        # all_actions = abs_6d_2_abs_euler(all_actions) 
        
        end_time = time.time()
        print("model cost time: ", end_time -start_time)
        inference_lock.acquire()
        inference_actions = all_actions
        if pre_action is None:
            pre_action = obs['eef_quat']

        inference_timestep = t
        inference_lock.release()
        return inference_actions #, obs['eef_6d'], status


def model_inference(args, config, ros_operator, save_episode=True):
    global inference_lock
    global inference_actions
    global inference_timestep
    global inference_thread
    
    policy = ClientModel("0.0.0.0", PORT)

    max_publish_step = config['episode_len']

    # Send initial positions
    left0 = init_qpos[:7]
    left1 = init_qpos[:7]
    right0 = init_qpos[7:]
    right1 = init_qpos[7:]
    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Enter any key to continue joint:")
    ros_operator.puppet_arm_publish_continuous(left1, right1)
    action = None
    # infer
    start_time = time.time()
    count = 0
    action_list = []
    proprio_list = []
    status_list = []
    
    with torch.inference_mode():
        policy.reset()

        t = 0
        # max_t = 0
        rate = rospy.Rate(args.publish_rate)
        bridge = CvBridge()
        frames_rgb = []

        def image_callback(msg):
            threading.Thread(target=process_image, args=(msg,)).start()

        def process_image(msg):
            img = bridge.imgmsg_to_cv2(msg, "bgr8")
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)
        rospy.Subscriber('/camera_f/color/image_raw', Image, image_callback)

        replayed_id = 1
        while t < max_publish_step and not rospy.is_shutdown():
            if current_key == 'q':
                print("Press q, break")
                break
            pre_action = action
            # try:
            step_start_time = time.time()
            # action, proprio, status = get_action(args, config, ros_operator, policy, t, pre_action)
            action = get_action(args, config, ros_operator, policy, t, pre_action)
            action_list.append(action)
            # proprio_list.append(proprio)
            # status_list.append(to_dict(status, t))
            # print("!!!"*88)
            duration = time.time() - start_time
            # while time.time() - step_start_time  1./30:
            #     print("sleep")
                
            count += 1
            # print("avg Hz:", count/duration)
            left_action = action[:7] 
            right_action = action[7:14]
            if SLEEP:
                time.sleep(0.05)
            # if replayed_id >= len(replay_data): exit(0)
            # ros_operator.puppet_arm_publish(replay_data[replayed_id][:7], replay_data[replayed_id][7:])
            # replayed_id += 1
            print('proprio', policy.pred_proprio)
            print('[left][right]', left_action, right_action)
            # input("Enter any key to continue joint:")
            ros_operator.puppet_arm_publish(left_action, right_action)  # joint publish
            # print("right action xyz:", right_action[:3])
            # print("right proprio xyz:", proprio[10:13])
            
            
            # print("error right:", np.sqrt(np.sum((right_action[:3] - proprio[10:13])**2)/3))
            # assert np.sqrt(np.sum((right_action[:3] - proprio[10:13])**2)/3) > 0.1
            if args.use_robot_base:
                vel_action = action[14:16]
                ros_operator.robot_base_publish(vel_action)
            t += 1
            # except:
                # continue
            rate.sleep()

        if current_key == 'q':
            print("\nProgram interrupted, processing...")
            while True:
                feedback = input("\nDo you think this inference was good or bad? (y/n): ").strip().lower()
                if feedback in ['y', 'n']:
                    break
                print("Please enter 'y' or 'n'")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if feedback == 'y':
                result_dir = "/home/agilex/data_processed/robotwin/infer_record/good"
            # else:
            #     result_dir = "/home/agilex/data_processed/robotwin/infer_record/bad"
                os.makedirs(result_dir, exist_ok=True)
                
                # time.sleep(1)
                # Save videos 
                safe_instruction = current_language_instruction.replace(" ", "_").replace("/", "-")
                video_path = os.path.join(result_dir, f"{safe_instruction}_{current_data_type}_{current_action_type}_{timestamp}_cameara_f.mp4")
                mediapy.write_video(video_path, frames_rgb, fps=30, codec='h264')
                print(f"Video saved to: {video_path}")
                
            exit(0)

            
        # np.save("action.npy", action_list)
        # np.save("proprio.npy", proprio_list)
        # print(status_list[0])
        # for status in status_list:
        #     with open('status.json', 'a+') as f:
        #         json.dump(status, f)
        #         f.write('\n')
            
def to_dict(piper_msg, t):
    return {
        'ctrl_mode': piper_msg.ctrl_mode,
        'arm_status': piper_msg.arm_status,
        'mode_feedback': piper_msg.mode_feedback,
        'teach_status': piper_msg.teach_status,
        'motion_status': piper_msg.motion_status,
        'trajectory_num': piper_msg.trajectory_num,
        'err_code': piper_msg.err_code,
        'joint_1_angle_limit': piper_msg.joint_1_angle_limit,
        'joint_2_angle_limit': piper_msg.joint_2_angle_limit,
        'joint_3_angle_limit': piper_msg.joint_3_angle_limit,
        'joint_4_angle_limit': piper_msg.joint_4_angle_limit,
        'joint_5_angle_limit': piper_msg.joint_5_angle_limit,
        'joint_6_angle_limit': piper_msg.joint_6_angle_limit,
        'communication_status_joint_1': piper_msg.communication_status_joint_1,
        'communication_status_joint_2': piper_msg.communication_status_joint_2,
        'communication_status_joint_3': piper_msg.communication_status_joint_3,
        'communication_status_joint_4': piper_msg.communication_status_joint_4,
        'communication_status_joint_5': piper_msg.communication_status_joint_5,
        'communication_status_joint_6': piper_msg.communication_status_joint_6,
        'time_step': t
    }

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='aloha_mobile_dummy', required=False)
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000, required=False)
    parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=False, required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--control_arm_left_pose_topic', action='store', type=str, help='control_arm_left_pose_topic',
                        default='/control/end_pose_left', required=False)
    parser.add_argument('--control_arm_right_pose_topic', action='store', type=str, help='control_arm_right_pose_topic',
                        default='/control/end_pose_right', required=False)

    # topic name of arm end pose
    parser.add_argument('--puppet_arm_left_pose_topic', action='store', type=str, help='puppet_arm_left_pose_topic',
                        default='/puppet/end_pose_left', required=False)
    parser.add_argument('--puppet_arm_right_pose_topic', action='store', type=str, help='puppet_arm_right_pose_topic',
                        default='/puppet/end_pose_right', required=False)
 
    # topic name of arm status
    parser.add_argument('--status_topic', action='store', type=str, help='puppet_arm_left_pose_topic',
                        default='/puppet/arm_status', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=15, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=0, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=30, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)

    parser.add_argument('--rel_a', action='store', type=bool, help='rel eef or abs eef',
                        default=False, required=False)
    args = parser.parse_args()
    return args

current_key = None

def keyboard_listener():
    global current_key
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == 'q':
                    current_key = 'q'
                    break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    config = {
        'episode_len': args.max_publish_step,
        'temporal_agg': args.temporal_agg,
        'camera_names': task_config['camera_names'],
    }
    keyboard_thread = threading.Thread(target=keyboard_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    model_inference(args, config, ros_operator, save_episode=True)


if __name__ == '__main__':
    main()
