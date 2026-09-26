import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.nnx import traversals
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.lora as _lora
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    # A loader that declares `converts_dtype` casts the checkpoint to another dtype, so its dtypes
    # intentionally differ from the float32 shapes the model was traced with. Only shapes compare.
    converts_dtype = getattr(loader, "converts_dtype", False)
    at.check_pytree_equality(
        expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=not converts_dtype
    )

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _norm_keypath(keypath: tuple[Any, ...]) -> tuple[Any, ...]:
    """Normalizes a state keypath so that keys from a checkpoint and from the model state compare equal.

    Sequence indices are integers in `nnx` keypaths but strings in the nested dicts read from a checkpoint.
    """
    normalized = []
    for key in keypath:
        try:
            normalized.append(int(key))
        except (TypeError, ValueError):
            normalized.append(key)
    return tuple(normalized)


def _create_params(config: _config.TrainConfig, params_shape: nnx.State, rng: at.KeyArrayLike) -> nnx.State:
    """Materializes the initial parameters in the dtype they will be trained in.

    pi0/pi0.5 models have ~3.4B parameters. Creating them with `config.model.create()` materializes all of
    them in float32 (~13 GB) on the accelerator before any optimizer state exists, which does not fit on
    smaller cards.

    The base parameters come from the weight loader, which is expected to have already placed them on the
    accelerator in bfloat16 (`DeviceCheckpointWeightLoader`), and only the trainable LoRA parameters are
    initialized here, in float32. A loader that returns host arrays instead is still supported: those are
    converted one at a time, so the float32 copy of the whole checkpoint is never resident at once.
    """
    loaded = {
        _norm_keypath(keypath): value
        for keypath, value in traversals.flatten_mapping(
            _load_weights_and_validate(config.weight_loader, params_shape.to_pure_dict())
        ).items()
    }
    keys = jax.random.split(rng, len(params_shape.flat_state()))

    converted = {}
    for (keypath, var), key in zip(params_shape.flat_state().items(), keys, strict=True):
        frozen = config.freeze_filter(keypath, var.value)
        if (norm_keypath := _norm_keypath(keypath)) in loaded:
            value = loaded.pop(norm_keypath)
            if isinstance(value, jax.Array):
                # `DeviceCheckpointWeightLoader` already placed this on the accelerator in its target dtype,
                # so copying it back to the host would only undo that.
                converted[keypath] = value
            else:
                converted[keypath] = np.asarray(value).astype(jnp.bfloat16 if frozen else jnp.float32, copy=False)
        elif frozen:
            raise ValueError(
                f"Frozen parameter {jax.tree_util.keystr(keypath)} is missing from the checkpoint loaded by "
                f"{config.weight_loader}. It cannot be initialized randomly because it is not trained."
            )
        else:
            # Parameters the checkpoint does not provide are the ones trained from scratch, i.e. the LoRA
            # adapters. They use the default initializer of `lora.LoRAConfig`.
            converted[keypath] = np.asarray(
                _lora.LoRAConfig(rank=1).init_fn(key, var.value.shape, jnp.float32), dtype=np.float32
            )

    params = params_shape
    params.replace_by_pure_dict(traversals.unflatten_mapping(converted))
    return params


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    # Trace the model to obtain its structure and parameter shapes without allocating any values.
    # See `_create_params` for why the model is not created on the accelerator.
    _, model_rng = jax.random.split(init_rng)
    model_def, params_shape = nnx.split(nnx.eval_shape(config.model.create, model_rng))

    def init(params: nnx.State) -> training_utils.TrainState:
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=model_def,
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, params_shape)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    _, params_rng = jax.random.split(init_rng)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    # Build the parameters, then place them and the optimizer state on the mesh. This is done eagerly
    # instead of under `jax.jit` so that the weights are not buffered twice on the accelerator.
    params = jax.device_put(_create_params(config, params_shape, params_rng), replicated_sharding)
    train_state = jax.device_put(init(params), state_sharding)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Initialize the train state before the data loader starts iterating, so that restoring the base
    # checkpoint does not have to share host memory with the loader's worker processes and prefetch
    # buffers. The data loader itself is only stepped after the state is fully restored.
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state.params)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
