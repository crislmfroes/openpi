"""Data transforms for the bimanual OpenArm dataset.

The dataset records two 7-DoF arms with a gripper each, so the state and action vectors are 16
dimensional: `[left_joint_1..7, left_gripper, right_joint_1..7, right_gripper]`. Actions are
absolute joint position commands (the measured state is the robot's feedback, the action is the
commanded target), so no delta transform is applied by default.

Three cameras are recorded: a chest-mounted view and one wrist camera per arm. They map onto the
three image slots the pi0/pi0.5 models expect.
"""

import dataclasses
from typing import ClassVar

import numpy as np
import torch

from openpi import transforms

# Maps the image slot expected by the model to the camera name in the dataset. `data["images"]` is keyed
# by the bare camera name, without the `observation.images.` prefix used in the LeRobot metadata.
CAMERA_MAP: dict[str, str] = {
    "base_0_rgb": "cam_chest",
    "left_wrist_0_rgb": "left_cam_wrist",
    "right_wrist_0_rgb": "right_cam_wrist",
}

# Number of state/action dimensions: 7 joints + 1 gripper per arm.
ROBOT_DIM = 16

# Boolean mask that selects the joint dimensions (and not the grippers) for delta actions:
# (True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, False)
DELTA_MASK: tuple[bool, ...] = (True,) * 7 + (False,) + (True,) * 7 + (False,)


def make_openarm_example() -> dict:
    """Creates a random input example for the OpenArm policy."""
    return {
        "state": np.ones((ROBOT_DIM,)),
        "images": {name: np.random.randint(256, size=(3, 224, 224), dtype=np.uint8) for name in CAMERA_MAP.values()},
        "prompt": "put the lego brick in the box",
    }


def _to_uint8_hwc(image) -> np.ndarray:
    """Normalize a camera image to uint8 with shape [height, width, channels].

    LeRobot hands back float32 images in [0, 1] with shape [channels, height, width]. The transforms
    downstream (`ResizeImages` and `Observation.from_dict`) expect uint8 images with shape
    [height, width, channels].
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if image.max() > 1.0 + 1e-6:
            # Already scaled to [-1, 1] by an upstream transform.
            image = (image + 1.0) / 2.0
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")
    if image.shape[0] <= 4 < image.shape[-1]:
        # [channels, height, width] -> [height, width, channels]
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class OpenArmInputs(transforms.DataTransformFn):
    """Inputs for the bimanual OpenArm policy.

    Expected inputs:
    - images: dict[name, img] where img is [channels, height, width] in [0, 1] or uint8
    - state: [16]
    - actions: [action_horizon, 16]
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = tuple(CAMERA_MAP.values())

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(f"Unexpected cameras {sorted(unexpected)}, expected {self.EXPECTED_CAMERAS}")

        images = {}
        image_masks = {}
        for dest, source in CAMERA_MAP.items():
            if source not in in_images:
                raise ValueError(f"Missing camera {source!r}, expected {self.EXPECTED_CAMERAS}")
            images[dest] = _to_uint8_hwc(in_images[source])
            image_masks[dest] = np.True_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"], dtype=np.float32),
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class OpenArmOutputs(transforms.DataTransformFn):
    """Outputs for the bimanual OpenArm policy.

    The model predicts 32-dimensional actions, of which only the first 16 are meaningful for this
    robot; the rest is padding.
    """

    action_dim: int = ROBOT_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
