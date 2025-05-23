import concurrent.futures
import copy
import glob
import multiprocessing
import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import zarr
import zarr.storage
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5

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


# convert raw hdf5 data to replay buffer, which is used for diffusion policy training
def _convert_real_to_dp_replay(
    store: zarr.storage.Store,
    shape_meta: dict,
    dataset_dir: Path,
    n_workers: Optional[int] = None,
    max_inflight_tasks: Optional[int] = None,
) -> ReplayBuffer:
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
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

    states = torch.load(dataset_dir / "states.pth")
    states = states.float().cpu().numpy()
    actions = torch.load(dataset_dir / "abs_actions.pth")
    actions = actions.float().cpu().numpy()
    rel_actions = torch.load(dataset_dir / "rel_actions.pth")
    rel_actions = rel_actions.float().cpu().numpy()

    with open(dataset_dir / "seq_lengths.pkl", "rb") as seq_len_file:
        seq_lengths = pickle.load(seq_len_file)

    n_rollout = len(list((dataset_dir / "obses").glob("*.mp4")))
    n = n_rollout

    states = states[:n]
    actions = actions[:n]
    rel_actions = rel_actions[:n]
    seq_lengths = seq_lengths[:n]
    proprios = states[..., :2].copy()  # For pusht, first 2 dim of states is proprio
    velocities = torch.load(dataset_dir / "velocities.pth")
    velocities = velocities[:n].float().cpu().numpy()
    states = np.concatenate([states, velocities], axis=-1)
    proprios = np.concatenate([proprios, velocities], axis=-1)
    print(f"Loaded {n} rollouts")

    all_steps = np.array(seq_lengths).sum()
    c, h, w = shape_meta["obs"]["image"]["shape"]
    rgb_data_dict["image"] = np.zeros((all_steps, h, w, c), dtype=np.uint8)
    lowdim_data_dict["action"] = np.zeros(
        (all_steps, actions.shape[-1]), dtype=np.float32
    )
    lowdim_data_dict["rel_action"] = np.zeros(
        (all_steps, rel_actions.shape[-1]), dtype=np.float32
    )
    lowdim_data_dict["proprios"] = np.zeros(
        (all_steps, proprios.shape[-1]), dtype=np.float32
    )
    for epi_idx in tqdm(range(0, n_rollout, 100), desc="Loading episodes"):
        end_idx = min(epi_idx + 100, n_rollout)
        hdf5_path = os.path.join(
            dataset_dir, "obses", f"episode_{epi_idx:03d}_to_{end_idx:03d}.hdf5"
        )
        data_dict, file_handler = load_dict_from_hdf5(hdf5_path)
        for j, epi_end in enumerate(data_dict["episode_ends"]):
            epi_end = int(epi_end)
            if j == 0:
                frames = data_dict["frames"][:epi_end]
            else:
                last_epi_end = data_dict["episode_ends"][j - 1]
                last_epi_end = int(last_epi_end)
                frames = data_dict["frames"][last_epi_end:epi_end]
            frames = np.stack([cv2.cvtColor(fm, cv2.COLOR_BGR2RGB) for fm in frames])
            frames = np.stack([center_crop(fm, (h, w)) for fm in frames])
            frames = np.stack(
                [cv2.resize(fm, (w, h), interpolation=cv2.INTER_AREA) for fm in frames]
            )
            # rgb_data_dict["image"].append(frames)
            seq_len = seq_lengths[epi_idx + j]
            # lowdim_data_dict["action"].append(actions[epi_idx + j, :seq_len])
            # lowdim_data_dict["proprios"].append(proprios[epi_idx + j, :seq_len])
            episode_end = prev_end + seq_lengths[epi_idx + j]
            rgb_data_dict["image"][prev_end:episode_end] = frames
            lowdim_data_dict["action"][prev_end:episode_end] = actions[
                epi_idx + j, :seq_len
            ]
            lowdim_data_dict["rel_action"][prev_end:episode_end] = rel_actions[
                epi_idx + j, :seq_len
            ]
            lowdim_data_dict["proprios"][prev_end:episode_end] = proprios[
                epi_idx + j, :seq_len
            ]
            prev_end = episode_end
            episode_ends.append(episode_end)
        file_handler.close()

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
            for hdf5_idx in tqdm(range(data.shape[0])):
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
                    executor.submit(img_copy, img_arr, zarr_idx, data, hdf5_idx)
                )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


class DINOWMPushTDataset(BaseImageDataset):
    """A dataset for the real-world data collected on Aloha robot."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        # assign config
        is_val = cfg.is_val
        if is_val:
            dataset_dir = Path(cfg.dataset_dir) / "val"
        else:
            dataset_dir = Path(cfg.dataset_dir) / "train"
        shape_meta = cfg.shape_meta
        horizon = (cfg.horizon + 1) * cfg.skip_frame
        pad_before = cfg.pad_before
        pad_after = cfg.pad_after
        use_cache = cfg.use_cache
        self.val_horizon = (
            (cfg.val_horizon + 1) * cfg.skip_frame if "val_horizon" in cfg else horizon
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
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.cfg = cfg
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.dataset_dir = dataset_dir
        self.skip_frame = cfg.skip_frame
        self.resolution = cfg.resolution

    def get_normalizer(self, mode: str = "none", **kwargs: dict) -> LinearNormalizer:
        """Return a normalizer for the dataset."""
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])
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
            elif key.endswith("proprios"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
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
        for key in self.lowdim_keys:
            obs_dict[key] = sample[key].astype(np.float32)
            obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            del sample[key]

        actions = sample["action"].astype(np.float32)
        rel_actions = sample["rel_action"].astype(np.float32)
        action_dim = actions.shape[-1]
        downsample_horizon = actions.shape[0] // self.skip_frame - 1
        action_len = downsample_horizon * self.skip_frame
        action_start = skip_start - self.skip_frame
        actions = actions[action_start : action_start + action_len]
        actions = actions.reshape(downsample_horizon, self.skip_frame * action_dim)
        rel_actions = rel_actions[action_start : action_start + action_len]
        rel_actions = rel_actions.reshape(
            downsample_horizon, self.skip_frame * action_dim
        )
        data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(actions),
            "rel_action": torch.from_numpy(rel_actions),
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return data

    def get_validation_dataset(self) -> "BaseImageDataset":
        """Return a validation dataset."""
        val_set_cfg = copy.deepcopy(self.cfg)
        val_set_cfg.is_val = True
        val_set = DINOWMPushTDataset(val_set_cfg)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.val_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
        )
        return val_set


def test_dataset() -> None:
    config_path = "/home/yixuan/diffusion-forcing/configurations/dataset/dinowm_pusht_dataset.yaml"  # noqa
    cfg = OmegaConf.load(config_path)
    cfg.dataset_dir = "/home/yixuan/dino_wm/data/pusht_noise"
    cfg.is_val = True
    cfg.skip_frame = 3
    dataset = DINOWMPushTDataset(cfg)
    dataset.get_normalizer()
    print(dataset[200])
    print("success!")
    val_dataset = dataset.get_validation_dataset()
    val_dataset.get_normalizer()
    print(val_dataset[200])
    print("success!")


if __name__ == "__main__":
    test_dataset()
