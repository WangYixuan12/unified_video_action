import concurrent.futures
import glob
import multiprocessing
import os
import shutil
from typing import Dict, Optional
import copy

import cv2
import numpy as np
import torch
import zarr
import zarr.storage
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from pytorch3d.transforms import matrix_to_rotation_6d
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
from yixuan_utilities.draw_utils import center_crop, resize_to_height

from unified_video_action.codecs.imagecodecs_numcodecs import Jpeg2k, register_codecs
from unified_video_action.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.common.replay_buffer import ReplayBuffer
from unified_video_action.common.sampler import (
    SequenceSampler,
    get_val_mask,
)
from unified_video_action.common.draw_utils import concat_img_h

from .base_dataset import BaseImageDataset

register_codecs()


def read_one_episode(
    dataset_path: str, h: int, w: int
) -> tuple[dict, dict, np.ndarray]:
    """read one episode from kaifeng's data folder

    Args:
        dataset_path (str): path to the dataset
        h (int): height of the image
        w (int): width of the image

    Returns:
        tuple[dict, dict, list]: rgb_data_dict, depth_data_dict, time_list, time_idx

    Folder structure:
    dataset_path
    ├── camera_0
    |   ├── rgb
    |   |   ├── 000000.jpg
    |   |   ├── ...
    |   ├── depth
    |   |   ├── 000000.png
    |   |   ├── ...
    |...
    ├── camera_n
    ├── timestamps.txt
    """
    camera_dirs = glob.glob(os.path.join(dataset_path, "camera_*"))
    rgb_data_dict: dict = dict()
    depth_data_dict: dict = dict()
    times = np.loadtxt(os.path.join(dataset_path, "timestamps.txt"))
    time_list = times[:, -4:]
    time_idx = times[:, :4]
    for camera_dir in sorted(camera_dirs):
        camera_name = os.path.basename(camera_dir)
        rgb_data_dict[camera_name] = []
        depth_data_dict[camera_name] = []
        rgb_dir = os.path.join(camera_dir, "rgb")
        depth_dir = os.path.join(camera_dir, "depth")
        rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
        depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
        assert len(rgb_paths) == len(depth_paths)
        for rgb_path, depth_path in zip(rgb_paths, depth_paths):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_crop = center_crop(rgb_img, (h, w))
            rgb_resize = cv2.resize(rgb_crop, (w, h), interpolation=cv2.INTER_AREA)
            rgb_img = rgb_resize
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth_crop = center_crop(depth_img, (h, w))
            depth_resize = cv2.resize(
                depth_crop, (w, h), interpolation=cv2.INTER_NEAREST
            )
            depth_img = depth_resize
            rgb_data_dict[camera_name].append(rgb_img)
            depth_data_dict[camera_name].append(depth_img)
        rgb_data_dict[camera_name] = np.stack(rgb_data_dict[camera_name], axis=0)
        depth_data_dict[camera_name] = np.stack(depth_data_dict[camera_name], axis=0)

    # sync obs to make sure each row corresponds to the same time
    # time_list: (n, k) - each row is (time_for_view_0, ..., time_for_view_k)
    # time_idx: (n, k) - each row is (idx_for_view_0, ..., idx_for_view_k)
    # rgb_data_dict: {view_name: (n, h, w, c)}
    # depth_data_dict: {view_name: (n, h, w, c)}
    time_idx = time_idx.astype(int)
    min_idx = time_idx.min()
    max_idx = time_idx.max()
    sync_time_list = []
    # sync_time_idx = []
    sync_rgb_data_dict: dict = {k: [] for k in rgb_data_dict.keys()}
    sync_depth_data_dict: dict = {k: [] for k in depth_data_dict.keys()}
    start_sync = False
    for i in range(min_idx, max_idx + 1):
        idx_in_each_view = np.zeros(time_idx.shape[1], dtype=int)
        incomplete = False
        for view_i in range(idx_in_each_view.shape[0]):
            while (
                idx_in_each_view[view_i] < time_idx.shape[0]
                and time_idx[idx_in_each_view[view_i], view_i] < i
            ):
                idx_in_each_view[view_i] += 1
            if idx_in_each_view[view_i] >= time_idx.shape[0]:
                start_sync = False
                incomplete = True
                break
            if time_idx[idx_in_each_view[view_i], view_i] != i:
                incomplete = True
                break
        if not incomplete and not start_sync:
            start_sync = True
        if start_sync:
            sync_time = []
            for view_i, t_idx_in_i in enumerate(idx_in_each_view):
                sync_time.append(time_list[t_idx_in_i, view_i])
                view_k = list(rgb_data_dict.keys())[view_i]
                sync_rgb_data_dict[view_k].append(rgb_data_dict[view_k][t_idx_in_i])
                sync_depth_data_dict[view_k].append(depth_data_dict[view_k][t_idx_in_i])
            # sync_time_idx.append(i)
            sync_time_list.append(np.array(sync_time))
    sync_time_np: np.ndarray = np.stack(sync_time_list, axis=0)
    sync_time_np = sync_time_np.mean(axis=1)
    return sync_rgb_data_dict, sync_depth_data_dict, sync_time_np


def read_robot_data(robot_data_path: str) -> tuple[list, list]:
    """read robot data from kaifeng's data folder

    Args:
        robot_data_path (str): path to the robot data

    Returns:
        list[tuple]: robot_data_list of (time, data)

    Folder structure:
    robot_data_path
    ├── 172xxxxxx.xxx.txt
    ├── 172xxxxxx.xxx.txt
    ├── ...
    where each txt file contains (x, y, z) or more, depending on the task
    """
    robot_data_list = list()
    time_list = list()
    robot_data_paths = sorted(glob.glob(os.path.join(robot_data_path, "*.txt")))
    for robot_data_path in robot_data_paths:
        time_str = os.path.basename(robot_data_path)[:-4]
        time = float(time_str)
        data = np.loadtxt(robot_data_path)
        robot_data_list.append(data)
        time_list.append(time)
    return robot_data_list, time_list


# convert raw hdf5 data to replay buffer, which is used for diffusion policy training
def _convert_real_to_dp_replay(
    store: zarr.storage.Store,
    shape_meta: dict,
    dataset_dir: str,
    n_workers: Optional[int] = None,
    max_inflight_tasks: Optional[int] = None,
) -> ReplayBuffer:
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    camera_keys = list()
    depth_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
            camera_keys.append(key[:-4])
        if type == "depth":
            depth_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    episodes_names = os.listdir(dataset_dir)
    episodes_idx = [int(name) for name in episodes_names if name.isdigit()]
    episodes_idx = sorted(episodes_idx)
    robot_data_path = os.path.join(dataset_dir, "robot")
    robot_data, robot_time = read_robot_data(robot_data_path)

    episode_ends = list()
    prev_end = 0
    lowdim_data_dict: dict = dict()
    rgb_data_dict: dict = dict()
    depth_data_dict: dict = dict()
    # episode-level loop
    for epi_idx in tqdm(episodes_idx, desc="Loading episodes"):
        dataset_path = os.path.join(dataset_dir, f"{epi_idx}")
        any_key = rgb_keys[0]  # assume shapes are the same
        any_shape = tuple(shape_meta["obs"][any_key]["shape"])
        _, h, w = any_shape
        async_rgb_data, async_depth_data, time_list = read_one_episode(
            dataset_path, h, w
        )

        # frame-level loop
        robot_data_idx = 0
        episode_length = 0
        for i in tqdm(range(time_list.shape[0]), desc="Loading frames"):
            while (
                robot_data_idx < len(robot_time)
                and robot_time[robot_data_idx] < time_list[i]
            ):
                robot_data_idx += 1
            if robot_data_idx >= len(robot_time):
                break
            episode_length += 1

            # save lowdim data to lowedim_data_dict
            if "action" not in lowdim_data_dict:
                lowdim_data_dict["action"] = list()
            lowdim_data_dict["action"].append(robot_data[robot_data_idx])

            # save rgb data to rgb_data_dict
            for key in rgb_keys:
                if key not in rgb_data_dict:
                    rgb_data_dict[key] = list()
                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape
                img = async_rgb_data[key[:-4]][i]
                rgb_data_dict[key].append(img)

            # save depth data to depth_data_dict
            for key in depth_keys:
                if key not in depth_data_dict:
                    depth_data_dict[key] = list()
                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape
                img = async_depth_data[key[:-6]][i]
                depth_data_dict[key].append(img)
                episode_length = time_list.shape[0]

        # manually slice the episode into two halves to save some data for training
        half_episode_length = episode_length // 2
        episode_end = prev_end + half_episode_length
        episode_ends.append(episode_end)
        episode_end = prev_end + episode_length
        episode_ends.append(episode_end)
        prev_end = episode_end

    lowdim_data_dict = {k: np.stack(v, axis=0) for k, v in lowdim_data_dict.items()}
    rgb_data_dict = {k: np.stack(v, axis=0) for k, v in rgb_data_dict.items()}
    depth_data_dict = {k: np.stack(v, axis=0) for k, v in depth_data_dict.items()}

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
            hdf5_arr = data
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
            hdf5_arr = data
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


class KFDataset(BaseImageDataset):
    """A dataset for the real-world data collected by Kaifeng."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        # assign config
        shape_meta = cfg.shape_meta
        dataset_dir = cfg.dataset_dir
        horizon = cfg.horizon * cfg.skip_frame
        pad_before = cfg.pad_before
        pad_after = cfg.pad_after
        use_cache = cfg.use_cache
        seed = cfg.seed
        val_ratio = cfg.val_ratio
        manual_val_mask = cfg.manual_val_mask if "manual_val_mask" in cfg else False
        manual_val_start = cfg.manual_val_start if "manual_val_start" in cfg else -1
        self.val_horizon = (
            cfg.val_horizon * cfg.skip_frame if "val_horizon" in cfg else horizon
        )
        self.skip_idx = cfg.skip_idx if "skip_idx" in cfg else 1

        replay_buffer = None
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
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.dataset_dir = dataset_dir
        self.skip_frame = cfg.skip_frame

    def get_normalizer(self, mode: str = "none", **kwargs: dict) -> LinearNormalizer:
        """Return a normalizer for the dataset."""
        normalizer = LinearNormalizer()

        # action
        if len(self.replay_buffer["action"].shape) == 3:
            # if it is controlled by GELLO
            t1 = self.replay_buffer["action"][:, 0]
            r1 = self.replay_buffer["action"][:, 1:4]
            t2 = self.replay_buffer["action"][:, 4]
            r2 = self.replay_buffer["action"][:, 5:8]
            g1 = self.replay_buffer["action"][:, 8, 0:1]
            g2 = self.replay_buffer["action"][:, 8, 1:2]
            r1_6d = matrix_to_rotation_6d(torch.from_numpy(r1))
            r1_6d_np = r1_6d.numpy()
            r2_6d = matrix_to_rotation_6d(torch.from_numpy(r2))
            r2_6d_np = r2_6d.numpy()
            action = np.concatenate([t1, r1_6d_np, g1, t2, r2_6d_np, g2], axis=-1)
        else:
            action = self.replay_buffer["action"]
        stat = array_to_stats(action)
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
        skip_start = np.random.randint(0, self.skip_frame)
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
        actions = actions[skip_start :: self.skip_frame]
        if len(actions.shape) == 3:
            # if it is controlled by GELLO
            t1 = actions[:, 0]  # (T, 3)
            r1 = actions[:, 1:4]  # (T, 3, 3)
            t2 = actions[:, 4]  # (T, 3)
            r2 = actions[:, 5:8]  # (T, 3, 3)
            g1 = actions[:, 8, 0:1]  # (T, 1)
            g2 = actions[:, 8, 1:2]  # (T, 1)
            r1_6d = matrix_to_rotation_6d(torch.from_numpy(r1))
            r1_6d_np = r1_6d.numpy()
            r2_6d = matrix_to_rotation_6d(torch.from_numpy(r2))
            r2_6d_np = r2_6d.numpy()
            actions = np.concatenate([t1, r1_6d_np, g1, t2, r2_6d_np, g2], axis=-1)

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
        "/home/yixuan/diffusion-forcing/configurations/dataset/kf_dataset.yaml"
    )
    cfg = OmegaConf.load(config_path)
    # cfg.dataset_dir = "data/kf_data/0114_cloth1/recording_1"
    # cfg.skip_frame = 2
    dataset = KFDataset(cfg)
    print(dataset[0])
    print("success!")


def visualize_kf_dataset() -> None:
    config_path = (
        "/home/yixuan/diffusion-forcing/configurations/dataset/kf_dataset.yaml"
    )
    cfg = OmegaConf.load(config_path)
    cfg.dataset_dir = "data/kf_data/0112_box/recording_1"
    dataset = KFDataset(cfg)
    replay_buffer = dataset.replay_buffer
    actions = replay_buffer["action"]
    camera_0_rgb = replay_buffer["camera_0_rgb"]
    episode_ends = replay_buffer.episode_ends
    
    
    if len(actions.shape) == 3:
        # if it is controlled by GELLO
        t1 = actions[:, 0]  # (T, 3)
        r1 = actions[:, 1:4]  # (T, 3, 3)
        t2 = actions[:, 4]  # (T, 3)
        r2 = actions[:, 5:8]  # (T, 3, 3)
        g1 = actions[:, 8, 0:1]  # (T, 1)
        g2 = actions[:, 8, 1:2]  # (T, 1)
        r1_6d = matrix_to_rotation_6d(torch.from_numpy(r1))
        r1_6d_np = r1_6d.numpy()
        r2_6d = matrix_to_rotation_6d(torch.from_numpy(r2))
        r2_6d_np = r2_6d.numpy()
        actions = np.concatenate([t1, r1_6d_np, g1, t2, r2_6d_np, g2], axis=-1)
    
    # real_time_plotter = RealTimePlotter(title="actions")
    plt.ion()
    
    for i in range(len(episode_ends)):
        if i == 0:
            s = 0
        else:
            s = episode_ends[i - 1]
        e = episode_ends[i]
        
        # plt
        fig, ax = plt.subplots()
        lines = [ax.plot([], [])[0] for _ in range(actions.shape[1])]
        ax.set_title("actions")
        ax.set_xlim(0, e - s)
        ax.set_ylim(-1.0, 1.0)
        ax.legend([f"action_{i}" for i in range(actions.shape[1])])
        fig.canvas.draw()
        plt.show(block=False)
        
        imgs = []
        for j in range(s, e):
            for l_i, l in enumerate(lines):
                l.set_xdata(np.arange(j-s))
                l.set_ydata(actions[s:j, l_i])
            fig.canvas.draw()
            action_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            action_plot = action_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            rgb_plot = camera_0_rgb[j]
            rgb_plot = resize_to_height(rgb_plot, action_plot.shape[0])
            img = concat_img_h([rgb_plot, action_plot])
            imgs.append(img)
        imageio.mimsave(f"kf_data_episode_{i}.mp4", np.stack(imgs), fps=25)
        
    
    print(len(dataset))
    data = dataset[0]
    print(data)
    print("success!")


if __name__ == "__main__":
    test_kf_dataset()
    # visualize_kf_dataset()
