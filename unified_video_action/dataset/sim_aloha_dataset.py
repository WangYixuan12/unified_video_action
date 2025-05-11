import concurrent.futures
import glob
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Dict, Optional
import copy

import cv2
import h5py
import imageio
import numpy as np
import torch
import zarr
import zarr.storage
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from pytorch3d.transforms import matrix_to_rotation_6d
from tqdm import tqdm
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.kinematics_helper import KinHelper

from unified_video_action.codecs.imagecodecs_numcodecs import Jpeg2k, register_codecs
from unified_video_action.model.common.normalizer import (
    LinearNormalizer,
    array_to_stats,
    get_hundred_times_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.common.replay_buffer import ReplayBuffer
from unified_video_action.common.sampler import SequenceSampler, get_val_mask

from .base_dataset import BaseImageDataset

register_codecs()


def _convert_ee_actions(
    raw_actions: np.ndarray, joint_actions: np.ndarray
) -> np.ndarray:
    T, S, _, _, _ = raw_actions.shape

    pos = raw_actions[:, :, 0, :3, 3]  # (T, S, 3)
    mat = raw_actions[:, :, 0, :3, :3]  # (T, S, 3, 3)
    rot = matrix_to_rotation_6d(torch.from_numpy(mat.reshape(-1, 3, 3)))
    rot = rot.reshape(T, S, -1)  # (T, S, 6)
    gripper = joint_actions[:, 6::7][..., None]  # (T, S)
    cvt_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
    cvt_actions = cvt_actions.reshape(T, -1)  # (T, S*10)
    return cvt_actions


def _convert_to_single_ee_actions(
    raw_actions: np.ndarray, joint_actions: np.ndarray
) -> np.ndarray:
    pos = raw_actions[:, 1, 0, :3, 3]  # (T, 3)
    gripper = joint_actions[:, 6 + 7][..., None]  # (T, 1)
    cvt_actions = np.concatenate([pos, gripper], axis=-1).astype(np.float32)  # (T, 4)
    return cvt_actions


def _convert_to_bimanual_push_actions(
    raw_actions: np.ndarray, joint_actions: np.ndarray, robot_bases: np.ndarray
) -> np.ndarray:
    T, S, _, _, _ = raw_actions.shape
    modified_raw_actions = np.zeros_like(raw_actions)
    for i in range(S):
        ee_pose = raw_actions[:, i, 0]  # (T, 4, 4)
        world_t_robot_base = robot_bases[i]  # (4, 4)
        world_t_robot_base = np.tile(world_t_robot_base[None], (T, 1, 1))
        ee_pose = np.linalg.inv(world_t_robot_base) @ ee_pose  # (T, 4, 4)
        theta = np.pi * 5.0 / 12.0
        ee_pose[:, :3, :3] = np.array(
            [
                [np.sin(theta), 0.0, np.cos(theta)],
                [0.0, 1.0, 0.0],
                [-np.cos(theta), 0.0, np.sin(theta)],
            ]
        )
        ee_pose[:, 0, 3] = np.clip(ee_pose[:, 0, 3], 0.25, 1.0)
        ee_pose[:, 1, 3] = np.clip(ee_pose[:, 1, 3], -0.25, 0.25)
        ee_pose[:, 2, 3] = 0.02
        ee_pose = world_t_robot_base @ ee_pose
        modified_raw_actions[:, i, 0] = ee_pose
    return modified_raw_actions[:, :, 0, :2, 3].reshape(T, 4)


# convert raw hdf5 data to replay buffer, which is used for diffusion policy training
def _convert_real_to_dp_replay(
    store: zarr.storage.Store,
    shape_meta: dict,
    dataset_dir: str,
    n_workers: Optional[int] = None,
    max_inflight_tasks: Optional[int] = None,
    action_mode: str = "bimanual_push",
) -> ReplayBuffer:
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    depth_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        if type == "depth":
            depth_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    episodes_paths = glob.glob(os.path.join(dataset_dir, "episode_*.hdf5"))
    episodes_stem_name = [Path(path).stem for path in episodes_paths]
    episodes_idx = [int(stem_name.split("_")[-1]) for stem_name in episodes_stem_name]
    episodes_idx = sorted(episodes_idx)

    episode_ends = list()
    prev_end = 0
    lowdim_data_dict: dict = dict()
    rgb_data_dict: dict = dict()
    depth_data_dict: dict = dict()
    for epi_idx in tqdm(episodes_idx, desc="Loading episodes"):
        dataset_path = os.path.join(dataset_dir, f"episode_{epi_idx}.hdf5")
        with h5py.File(dataset_path) as file:
            # count total steps
            episode_length = file["ee_action"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)

            # save lowdim data to lowedim_data_dict
            for key in [*lowdim_keys, "action"]:
                data_key = "obs/" + key
                if key == "action":
                    data_key = (
                        "ee_action"
                        if "key" not in shape_meta["action"]
                        else shape_meta["action"]["key"]
                    )
                if key not in lowdim_data_dict:
                    lowdim_data_dict[key] = list()
                this_data = file[data_key][()]
                if key == "action" and data_key == "ee_action":
                    # this_data = _convert_ee_actions(
                    #     raw_actions=this_data,
                    #     joint_actions=file["joint_action"][()],
                    # )
                    if action_mode == "bimanual_push":
                        this_data = _convert_to_bimanual_push_actions(
                            raw_actions=this_data,
                            joint_actions=file["joint_action"][()],
                            robot_bases=file["robot_bases"][0],
                        )
                    elif action_mode == "single_ee":
                        this_data = _convert_to_single_ee_actions(
                            raw_actions=this_data,
                            joint_actions=file["joint_action"][()],
                        )
                    assert this_data.shape[0] == episode_length
                    assert this_data.shape[1:] == shape_meta["action"]["shape"]
                elif key == "action" and data_key == "joint_action":
                    assert this_data.shape[0] == episode_length
                    assert this_data.shape[1:] == shape_meta["action"]["shape"]
                else:
                    assert this_data.shape[0] == episode_length
                    assert this_data.shape[1:] == shape_meta["obs"][key]["shape"]
                lowdim_data_dict[key].append(this_data)

            for key in rgb_keys:
                if key not in rgb_data_dict:
                    rgb_data_dict[key] = list()
                imgs = file["obs"]["images"][key][()]
                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape
                crop_imgs = [center_crop(img, (h, w)) for img in imgs]
                resize_imgs = [
                    cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    for img in crop_imgs
                ]
                imgs = np.stack(resize_imgs, axis=0)
                assert imgs[0].shape == (h, w, c)
                rgb_data_dict[key].append(imgs)

            for key in depth_keys:
                if key not in depth_data_dict:
                    depth_data_dict[key] = list()
                imgs = file["obs"]["images"][key][()]
                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape
                crop_imgs = [center_crop(img, (h, w)) for img in imgs]
                resize_imgs = [
                    cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    for img in crop_imgs
                ]
                imgs = np.stack(resize_imgs, axis=0)[..., None]
                imgs = np.clip(imgs, 0, 1000).astype(np.uint16)
                assert imgs[0].shape == (h, w, c)
                depth_data_dict[key].append(imgs)

    def img_copy(
        zarr_arr: zarr.Array, zarr_idx: int, hdf5_arr: np.ndarray, hdf5_idx: int
    ) -> bool:
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception:
            return False

    # dump data_dict
    print("Dumping meta data")
    n_steps = episode_ends[-1]
    _ = meta_group.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    print("Dumping lowdim data")
    for key, data in lowdim_data_dict.items():
        data = np.concatenate(data, axis=0)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=data.shape,
            compressor=None,
            dtype=data.dtype,
        )

    print("Dumping rgb data")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures: set = set()
        for key, data in rgb_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta["obs"][key]["shape"])
            c, h, w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=this_compressor,
                dtype=np.uint8,
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx)
                )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    print("Dumping depth data")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in depth_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta["obs"][key]["shape"])
            c, h, w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=this_compressor,
                dtype=np.uint16,
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx)
                )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


class SimAlohaDataset(BaseImageDataset):
    """A dataset for the real-world data collected on Aloha robot."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        # assign config
        shape_meta = cfg.shape_meta
        dataset_dir = cfg.dataset_dir
        horizon = (cfg.horizon + 1) * cfg.skip_frame
        pad_before = cfg.pad_before
        pad_after = cfg.pad_after
        use_cache = cfg.use_cache
        seed = cfg.seed
        val_ratio = cfg.val_ratio
        manual_val_mask = cfg.manual_val_mask if "manual_val_mask" in cfg else False
        manual_val_start = cfg.manual_val_start if "manual_val_start" in cfg else -1
        self.val_horizon = (
            (cfg.val_horizon + 1) * cfg.skip_frame if "val_horizon" in cfg else horizon
        )
        self.skip_idx = cfg.skip_idx if "skip_idx" in cfg else 1
        self.action_mode = cfg.action_mode

        replay_buffer = None
        with h5py.File(f"{dataset_dir}/episode_0.hdf5") as file:
            self.robot_bases = file["robot_bases"][0].copy()
        if use_cache:
            cache_info_str = ""
            obs_shape_meta = shape_meta["obs"]
            for _, attr in obs_shape_meta.items():
                type = attr.get("type", "low_dim")
            cache_zarr_path = os.path.join(
                dataset_dir, f"cache{cache_info_str}.zarr.zip"
            )
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print("Cache does not exist. Creating!")
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_real_to_dp_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_dir=dataset_dir,
                            action_mode=self.action_mode,
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("Loaded!")
        else:
            replay_buffer = _convert_real_to_dp_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_dir=dataset_dir,
                action_mode=self.action_mode,
            )
        self.replay_buffer = replay_buffer

        rgb_keys = list()
        depth_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "depth":
                depth_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        if not manual_val_mask:
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
            )
        else:
            assert manual_val_start >= 0, "manual_val_start must be >= 0"
            assert (
                manual_val_start < replay_buffer.n_episodes
            ), "manual_val_start too large"
            val_mask = np.zeros((replay_buffer.n_episodes,), dtype=np.bool)
            val_mask[manual_val_start:] = True
        train_mask = ~val_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.downsample_horizon = cfg.horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.dataset_dir = dataset_dir
        self.skip_frame = cfg.skip_frame
        self.delta_action = cfg.delta_action
        if self.delta_action:
            self.kin_helper = KinHelper("trossen_vx300s")

    def get_normalizer(self, mode: str = "none", **kwargs: dict) -> LinearNormalizer:
        """Return a normalizer for the dataset."""
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])
        if self.delta_action:
            this_normalizer = get_hundred_times_normalizer_from_stat(stat)
        else:
            this_normalizer = get_range_normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith("pos"):
                # this_normalizer = get_range_normalizer_from_stat(stat)
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("vel"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        for key in self.depth_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        obs_dict = dict()
        skip_start = np.random.randint(0, self.skip_frame) + self.skip_frame
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(sample[key], -1, 1).astype(np.float32) / 255.0
            obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            # T,C,H,W
            del sample[key]
        for key in self.depth_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint16 image to float32
            obs_dict[key] = np.moveaxis(sample[key], -1, 1).astype(np.float32) / 1000.0
            obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            # T,C,H,W
            del sample[key]
        for key in self.lowdim_keys:
            obs_dict[key] = sample[key].astype(np.float32)
            obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            del sample[key]

        actions = sample["action"].astype(np.float32)
        action_dim = actions.shape[-1]
        downsample_horizon = actions.shape[0] // self.skip_frame - 1
        action_len = downsample_horizon * self.skip_frame
        action_start = skip_start - self.skip_frame
        actions = actions[action_start : action_start + action_len]
        actions = actions.reshape(downsample_horizon, self.skip_frame, action_dim)
        if self.delta_action:
            joint_pos = obs_dict["joint_pos"].astype(np.float32)
            robot_bases = self.robot_bases
            # compute ee pos in robot_bases[0]
            num_robot = joint_pos.shape[1] // 7
            world_t_ee_pose = np.zeros((joint_pos.shape[0], num_robot, 4, 4))
            for i in range(num_robot):
                for t in range(joint_pos.shape[0]):
                    joint_fk = np.concatenate(
                        [
                            joint_pos[t, i * 7 : (i + 1) * 7],
                            joint_pos[t, i * 7 + 6 : (i + 1) * 7],
                        ]
                    )
                    ee_pose = self.kin_helper.compute_fk_from_link_idx(
                        joint_fk, [self.kin_helper.sapien_eef_idx]
                    )[0]
                    world_t_ee_pose[t, i] = robot_bases[i] @ ee_pose
            if self.action_mode == "bimanual_push":
                d_actions = np.zeros_like(actions)
                d_actions[..., :2] = actions[..., :2] - world_t_ee_pose[:, 0:1, :2, 3]
                d_actions[..., 2:] = actions[..., 2:] - world_t_ee_pose[:, 1:2, :2, 3]
                actions = d_actions
            elif self.action_mode == "single_ee":
                d_actions = np.zeros_like(actions)
                d_actions[..., :3] = actions[..., :3] - world_t_ee_pose[:, 1:2, :3, 3]
                d_actions[..., 3:4] = (
                    actions[..., 3:4] - joint_pos[:, 13:14][:, None]
                ) / 100
            else:
                raise NotImplementedError
        actions = actions.reshape(downsample_horizon, self.skip_frame * action_dim)
        data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(actions),
        }
        return data

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return data


def test_kf_dataset() -> None:
    config_path = (
        "/home/yixuan/diffusion-forcing/configurations/dataset/sim_aloha_dataset.yaml"
    )
    cfg = OmegaConf.load(config_path)
    # cfg.dataset_dir = "/media/yixuan/Extreme SSD/projects/diffusion-forcing/data/sim_aloha/single_arm_transfer_cube_0407_v3"  # noqa
    # cfg.dataset_dir = "/media/yixuan/Extreme SSD/projects/diffusion-forcing/data/sim_aloha/pusht_test_0414"  # noqa
    cfg.dataset_dir = "/media/yixuan/Extreme SSD/projects/diffusion-forcing/data/sim_aloha/pusht_0407"  # noqa
    cfg.delta_action = False
    cfg.action_mode = "bimanual_push"
    cfg.shape_meta.action.shape = (4,)
    cfg.skip_frame = 3
    dataset = SimAlohaDataset(cfg)
    dataset.get_normalizer()
    print(dataset[200])
    print("success!")


def visualize_kf_dataset() -> None:
    config_path = (
        "/home/yixuan/diffusion-forcing/configurations/dataset/sim_aloha_dataset.yaml"
    )
    cfg = OmegaConf.load(config_path)
    dataset = SimAlohaDataset(cfg)
    replay_buffer = dataset.replay_buffer
    camera_0_rgb = replay_buffer["teleoperator_pov"]
    episode_ends = replay_buffer.episode_ends

    for i in range(len(episode_ends)):
        if i == 0:
            s = 0
        else:
            s = episode_ends[i - 1]
        e = episode_ends[i]

        imgs = camera_0_rgb[s:e]
        imageio.mimsave(f"sim_aloha_data_episode_{i}.mp4", np.stack(imgs), fps=25)

    print(len(dataset))
    data = dataset[0]
    print(data)
    print("success!")


if __name__ == "__main__":
    test_kf_dataset()
    # visualize_kf_dataset()
