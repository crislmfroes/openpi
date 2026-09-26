#!/usr/bin/env python3
"""Convert a LeRobot v3.0 dataset into the LeRobot v2.1 layout.

openpi pins a LeRobot revision that only understands the v2.1 dataset layout, which stores one
parquet file and one mp4 file per episode. Datasets recorded with a newer LeRobot use the v3.0
layout instead: consolidated `data/chunk-*/file-*.parquet` files, consolidated
`videos/<key>/chunk-*/file-*.mp4` files, `meta/tasks.parquet` and `meta/episodes/chunk-*/file-*.parquet`.

This script re-writes such a dataset as v2.1 by replaying every frame through LeRobot's own dataset
writer, so the resulting metadata is exactly what LeRobot expects. Videos are re-encoded, since the
v2.1 layout requires one self-contained mp4 per episode.

Usage:
    uv run examples/openarm/convert_v3_to_lerobot_v21.py \
        --src /home/fbot/openarm_bimanual_vr \
        --dst /home/fbot/openarm_bimanual_vr_v21
"""

import argparse
import functools
import json
import pathlib
import time

import av
import numpy as np
import pandas as pd

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from lerobot.common.datasets import video_utils

NON_VIDEO_KEYS_TO_DROP = ("video.g", "video.crf", "video.preset", "video.fast_decode", "is_depth_map")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Path to the source v3.0 LeRobot dataset.")
    parser.add_argument("--dst", required=True, help="Path to the destination v2.1 LeRobot dataset (must not exist).")
    parser.add_argument(
        "--vcodec",
        default="h264",
        choices=["h264", "hevc", "libsvtav1"],
        help="Video codec for the re-encoded episodes. h264 (libx264) is much faster to encode than libsvtav1.",
    )
    parser.add_argument("--crf", type=int, default=23, help="Constant rate factor. Lower means better quality.")
    parser.add_argument("--episodes", type=int, default=None, help="Optionally convert only the first N episodes.")
    return parser.parse_args()


def build_features(info: dict) -> dict:
    """Build a v2.1 feature dict from the v3.0 info.json features.

    LeRobot v2.1 validates frame shapes with `actual_shape != expected_shape`, so the declared shape
    has to be a tuple (a list never compares equal to `np.ndarray.shape`).
    """
    features = {}
    for key, feat in info["features"].items():
        if feat["dtype"] != "video":
            features[key] = {
                "dtype": feat["dtype"],
                "shape": tuple(feat["shape"]),
                "names": feat["names"],
            }
            continue
        vinfo = feat["info"]
        features[key] = {
            "dtype": "video",
            "shape": tuple(feat["shape"]),
            "names": feat["names"],
            "info": {
                "video.height": vinfo["video.height"],
                "video.width": vinfo["video.width"],
                "video.pix_fmt": vinfo["video.pix_fmt"],
                "video.fps": vinfo["video.fps"],
                "video.channels": vinfo["video.channels"],
                "has_audio": vinfo["has_audio"],
                "video.is_depth_map": False,
            },
        }
    return features


def iter_frames(paths: list[pathlib.Path]):
    """Yield RGB frames from one or more video files, in order."""
    for path in paths:
        with av.open(str(path)) as container:
            for frame in container.decode(container.streams.video[0]):
                yield frame.to_ndarray(format="rgb24")


def video_paths_for_key(src: pathlib.Path, info: dict, key: str) -> list[pathlib.Path]:
    """Return the consolidated video files for a camera, in playback order."""
    template = info["video_path"]
    found = []
    for chunk_dir in sorted((src / "videos" / key).glob("chunk-*")):
        for video_file in sorted(chunk_dir.glob("file-*.mp4")):
            chunk_index = int(chunk_dir.name.split("-")[1])
            file_index = int(video_file.name.split("-")[1].split(".")[0])
            found.append((chunk_index, file_index, video_file))
    if not found:
        raise SystemExit(f"No video files found for {key} under {src / 'videos' / key} (template {template}).")
    return [path for _, _, path in sorted(found)]


def main():
    args = parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    if dst.exists():
        raise SystemExit(f"Destination {dst} already exists. Remove it or choose another path.")

    info = json.loads((src / "meta" / "info.json").read_text())
    if info["codebase_version"] != "v3.0":
        print(f"WARNING: source declares codebase_version={info['codebase_version']}, expected v3.0.")

    fps = info["fps"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    print(f"Found {len(video_keys)} video streams: {video_keys}")

    data_files = sorted(src.glob("data/chunk-*/file-*.parquet"))
    episode_files = sorted((src / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not data_files or not episode_files:
        raise SystemExit("Could not find the v3.0 data/ or meta/episodes/ parquet files.")

    frames = pd.concat([pd.read_parquet(p) for p in data_files], ignore_index=True)
    episodes = pd.concat([pd.read_parquet(p) for p in episode_files], ignore_index=True)
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    frames = frames.sort_values("index").reset_index(drop=True)
    if args.episodes is not None:
        keep = set(episodes["episode_index"].to_numpy()[: args.episodes].tolist())
        episodes = episodes[episodes["episode_index"].isin(keep)].reset_index(drop=True)
        frames = frames[frames["episode_index"].isin(keep)].reset_index(drop=True)

    # Verify the consolidated videos line up with the episode boundaries before decoding anything.
    lengths = episodes["length"].to_numpy()
    expected_starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    for key in video_keys:
        from_ts = episodes[f"videos/{key}/from_timestamp"].to_numpy(dtype=float)
        starts = np.round(from_ts * fps).astype(int)
        if not np.array_equal(starts, expected_starts):
            raise SystemExit(f"Episode boundaries for {key} do not match the frame layout; cannot convert safely.")

    # LeRobot's writer hardcodes libsvtav1, which is very slow, so override the encoder.
    av.logging.set_level(av.logging.ERROR)
    original_encode = video_utils.encode_video_frames
    patched_encode = functools.partial(original_encode, vcodec=args.vcodec, crf=args.crf)
    video_utils.encode_video_frames = patched_encode
    lerobot_dataset.encode_video_frames = patched_encode

    dst.parent.mkdir(parents=True, exist_ok=True)
    dataset = lerobot_dataset.LeRobotDataset.create(
        repo_id=dst.name,
        fps=fps,
        root=dst,
        robot_type=info.get("robot_type") or "unknown",
        features=build_features(info),
        use_videos=True,
        image_writer_threads=8,
    )

    states = np.stack(frames["observation.state"].to_numpy()).astype(np.float32)
    actions = np.stack(frames["action"].to_numpy()).astype(np.float32)

    # One reader per camera, all stepping forward in lockstep.
    readers = {key: iter_frames(video_paths_for_key(src, info, key)) for key in video_keys}

    start_time = time.time()
    total = 0
    for row in episodes.itertuples(index=False):
        ep_index = int(row.episode_index)
        length = int(row.length)
        task = row.tasks[0] if isinstance(row.tasks, list | np.ndarray) else row.tasks

        for i in range(length):
            frame = {"observation.state": states[total + i], "action": actions[total + i], "task": task}
            for key, reader in readers.items():
                frame[key] = next(reader)
            dataset.add_frame(frame)

        dataset.save_episode()
        total += length
        done = ep_index - int(episodes["episode_index"].iloc[0]) + 1
        elapsed = time.time() - start_time
        eta = elapsed / done * (len(episodes) - done)
        print(f"[{done}/{len(episodes)}] {task!r} frames={total} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print(f"Done: {total} frames, {len(episodes)} episodes -> {dst}")


if __name__ == "__main__":
    main()
