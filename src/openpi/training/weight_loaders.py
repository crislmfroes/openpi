import dataclasses
import logging
import pathlib
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class DeviceCheckpointWeightLoader(WeightLoader):
    """Loads checkpoint weights straight onto the accelerator, in the dtype they are trained in.

    `CheckpointWeightLoader` restores everything into host memory as numpy arrays. The released pi0/pi0.5
    checkpoints are float32, so that materializes ~13 GB of host memory before the training code gets to
    cast the frozen parameters to bfloat16, and it peaks around 21 GB while reading. That does not leave
    room for the data loader, and the process gets OOM-killed.

    This loader avoids the host copy altogether:

    * `restore_type=jax.Array` makes Orbax read each parameter and place it on the device itself, so the
      float32 checkpoint is never fully resident in host memory.
    * `dtype` makes the conversion to bfloat16 happen inside the read, so the float32 form is only ever
      held one read at a time.
    * `restore_concurrent_gb` caps how many GB Orbax reads concurrently. Without it, Orbax starts every
      read at once and the peak is ~18 GB; capping it brings the peak down to ~13 GB at the same speed.

    Caveats found while measuring this on a 4070 with 23 GB of RAM:

    * `restore_type=np.ndarray` together with `dtype` gets OOM-killed, because numpy has no way to receive
      the converted values incrementally. Only the device restore is safe.
    * The cap has a hard floor: a single read can be up to ~2.4 GB, and a smaller cap raises
      "Requested more bytes than we reserved space for". 3 GB is the lowest value that works.
    """

    params_path: str
    # Frozen parameters are trained in bfloat16, so the checkpoint is converted during the read instead of
    # after it. This halves both the bytes transferred and the device memory the parameters occupy.
    dtype: str = "bfloat16"
    # Must stay above the largest single read (~2.4 GB), see the caveats above.
    restore_concurrent_gb: int = 3

    # `_load_weights_and_validate` uses this to know that the returned dtype is intentional and that only
    # the shapes of the loaded parameters are comparable to the shapes the model was traced with.
    converts_dtype: bool = True

    def load(self, params: at.Params) -> at.Params:
        import orbax.checkpoint as ocp

        params_path = pathlib.Path(download.maybe_download(self.params_path)).resolve()
        ckptr = ocp.PyTreeCheckpointer()
        if self.restore_concurrent_gb:
            # `PyTreeCheckpointHandler` takes the cap, but `PyTreeCheckpointer` does not forward it, so the
            # handler it built by default is replaced with one that has it.
            handler = ocp.PyTreeCheckpointHandler(restore_concurrent_gb=self.restore_concurrent_gb)
            ckptr._handler = handler  # noqa: SLF001

        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}
        loaded_params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree_util.tree_map(
                    lambda _: ocp.ArrayRestoreArgs(restore_type=jax.Array, dtype=getattr(jnp, self.dtype)),
                    item,
                ),
            ),
        )["params"]
        # `restore_params` also drops the "value" level that `nnx.State` adds when the checkpoint was
        # written by `save_state` during openpi training.
        flat_params = flax.traverse_util.flatten_dict(loaded_params)
        if all(kp[-1] == "value" for kp in flat_params):
            flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
        loaded_params = flax.traverse_util.unflatten_dict(flat_params)

        # The LoRA adapters are not in the base checkpoint and keep the reference values.
        return _merge_without_casting(loaded_params, params, missing_regex=r".*lora.*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_without_casting(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Like `_merge_params`, but keeps the dtype of the loaded parameters.

    `_merge_params` casts every loaded value to the dtype of the reference, which for a base checkpoint is
    float32. That undoes the point of restoring in bfloat16, so `DeviceCheckpointWeightLoader` uses this
    variant and only takes reference values for the keys the checkpoint does not contain.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    result = {k: v for k, v in flax.traverse_util.flatten_dict(loaded_params, sep="/").items() if k in flat_ref}

    missing = set(flat_ref) - set(result)
    unexpected = {k for k in missing if not re.fullmatch(missing_regex, k)}
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} model parameters are missing from the checkpoint, e.g. "
            f"{sorted(unexpected)[:5]}. They would be left randomly initialized."
        )
    for k in missing:
        result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
