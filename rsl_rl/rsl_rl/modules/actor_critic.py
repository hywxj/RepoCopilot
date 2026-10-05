# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        own_state = super().state_dict()
        adapted_state_dict = dict(state_dict)
        expanded_inputs = []

        actor_history_length = None
        loaded_actor_cols = None
        inserted_obs_per_frame = 4
        if "actor.0.weight" in adapted_state_dict and "actor.0.weight" in own_state:
            loaded_actor_cols = adapted_state_dict["actor.0.weight"].shape[1]
            current_actor_cols = own_state["actor.0.weight"].shape[1]
            actor_col_delta = current_actor_cols - loaded_actor_cols
            if actor_col_delta > 0 and actor_col_delta % inserted_obs_per_frame == 0:
                actor_history_length = actor_col_delta // inserted_obs_per_frame

        def expand_repeated_history_input(loaded_weight, current_weight, history_length, tail_dim=0):
            if history_length is None:
                return None
            if loaded_weight.shape[1] % history_length != 0 or current_weight.shape[1] % history_length != 0:
                return None
            loaded_step_dim = loaded_weight.shape[1] // history_length
            current_step_dim = current_weight.shape[1] // history_length
            if loaded_step_dim + inserted_obs_per_frame != current_step_dim:
                return None
            if tail_dim < 0 or tail_dim >= loaded_step_dim:
                return None

            loaded_main_dim = loaded_step_dim - tail_dim
            merged_weight = current_weight.clone()
            for frame_idx in range(history_length):
                loaded_start = frame_idx * loaded_step_dim
                current_start = frame_idx * current_step_dim
                merged_weight[:, current_start : current_start + loaded_main_dim] = loaded_weight[
                    :, loaded_start : loaded_start + loaded_main_dim
                ]
                merged_weight[
                    :, current_start + loaded_main_dim : current_start + loaded_main_dim + inserted_obs_per_frame
                ] = 0.0
                if tail_dim:
                    merged_weight[:, current_start + loaded_main_dim + inserted_obs_per_frame : current_start + current_step_dim] = loaded_weight[
                        :, loaded_start + loaded_main_dim : loaded_start + loaded_step_dim
                    ]
            return merged_weight

        for key in ("actor.0.weight", "critic.0.weight"):
            if key not in adapted_state_dict or key not in own_state:
                continue
            loaded_weight = adapted_state_dict[key]
            current_weight = own_state[key]
            if loaded_weight.shape == current_weight.shape:
                continue
            can_expand_input = (
                loaded_weight.ndim == 2
                and current_weight.ndim == 2
                and loaded_weight.shape[0] == current_weight.shape[0]
                and loaded_weight.shape[1] < current_weight.shape[1]
            )
            if not can_expand_input:
                continue

            tail_dim = 0
            if key == "critic.0.weight" and actor_history_length is not None:
                loaded_actor_step_dim = loaded_actor_cols // actor_history_length
                loaded_critic_step_dim = loaded_weight.shape[1] // actor_history_length
                tail_dim = loaded_critic_step_dim - loaded_actor_step_dim

            merged_weight = expand_repeated_history_input(
                loaded_weight, current_weight, actor_history_length, tail_dim=tail_dim
            )
            if merged_weight is None:
                merged_weight = current_weight.clone()
                merged_weight[:, : loaded_weight.shape[1]] = loaded_weight
                merged_weight[:, loaded_weight.shape[1] :] = 0.0

            adapted_state_dict[key] = merged_weight
            expanded_inputs.append((key, loaded_weight.shape, current_weight.shape))

        if expanded_inputs:
            print(f"[INFO] Expanded observation input layers while loading checkpoint: {expanded_inputs}")

        super().load_state_dict(adapted_state_dict, strict=strict)
        return True


class _GatedResidualActor(nn.Module):
    """Frozen blind actor with a gated correction or an independent stair actor."""

    def __init__(
        self,
        base_actor: nn.Module,
        residual_actor: nn.Module,
        base_actor_obs_dim: int,
        gate_obs_index: int,
        residual_scale: float,
        residual_action_scales: list[float] | None = None,
        active_base_action_scale: float = 1.0,
        squash_residual: bool = True,
        residual_observation_scales: list[float] | None = None,
        stopping_phase_index: int = -1,
    ):
        super().__init__()
        self.base_actor = base_actor
        self.residual_actor = residual_actor
        self.base_actor_obs_dim = base_actor_obs_dim
        self.gate_obs_index = gate_obs_index
        self.residual_scale = residual_scale
        self.active_base_action_scale = active_base_action_scale
        self.squash_residual = squash_residual
        self.register_buffer("stopping_phase_index", torch.tensor(stopping_phase_index, dtype=torch.long))
        input_dim = residual_actor[0].in_features
        if residual_observation_scales is None:
            residual_observation_scales = [1.] * input_dim
        if (len(residual_observation_scales) != input_dim or
                not torch.isfinite(torch.tensor(residual_observation_scales)).all() or
                any(scale <= 0 for scale in residual_observation_scales)):
            raise ValueError("residual_observation_scales must have one finite positive value per observation.")
        self.register_buffer("residual_observation_scales", torch.tensor(residual_observation_scales).float().unsqueeze(0))
        if residual_action_scales is None:
            residual_action_scales = [1.0] * residual_actor[-1].out_features
        self.register_buffer(
            "residual_action_scales",
            torch.tensor(residual_action_scales, dtype=torch.float).unsqueeze(0),
            persistent=False,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        base_action = self.base_actor(observations[:, : self.base_actor_obs_dim])
        gate_index = self.gate_obs_index
        if gate_index < 0:
            gate_index += observations.shape[-1]
        gate = observations[:, gate_index : gate_index + 1].abs().clamp(0.0, 1.0)
        residual_observations = observations*self.residual_observation_scales
        stop_index = int(self.stopping_phase_index)
        if stop_index >= 0:
            stopping = observations[:, stop_index:stop_index+1] > .5
            residual_observations = torch.cat((
                residual_observations[:, :self.base_actor_obs_dim],
                torch.where(stopping, 0., residual_observations[:, self.base_actor_obs_dim:])), dim=1)
        residual = self.residual_actor(residual_observations)
        if self.squash_residual:
            residual = torch.tanh(residual)
        residual = residual*self.residual_scale*self.residual_action_scales
        applied_residual = gate * residual
        self.last_gate = gate.detach()
        self.last_base_action = base_action.detach()
        self.last_applied_residual = applied_residual.detach()
        base_scale = 1.0-gate*(1.0-self.active_base_action_scale)
        return base_scale*base_action + applied_residual


class GatedResidualActorCritic(ActorCritic):
    """Preserve a blind actor and learn a gated stair action branch.

    The last actor observation is expected to contain the stair state machine
    value: -1 for descending, 0 for blind locomotion, and +1 for ascending,
    with fractional values permitted during transitions. Defaults retain the
    bounded residual mode; optional blind initialization copies a full actor.
    """

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        residual_hidden_dims=[256, 128],
        activation="elu",
        init_noise_std=0.15,
        noise_std_type: str = "scalar",
        base_actor_obs_dim: int = 960,
        gate_obs_index: int = -1,
        residual_scale: float = 0.25,
        residual_action_scales: list[float] | None = None,
        active_base_action_scale: float = 1.0,
        squash_residual: bool = True,
        initialize_stair_actor_from_blind: bool = False,
        min_action_std: float = 0.05,
        max_action_std: float = 0.25,
        residual_observation_scales: list[float] | None = None,
        low_noise_obs_indices: list[int] | None = None,
        low_noise_scale: float = 1.0,
        train_stair_conditioning_only: bool = False,
        **kwargs,
    ):
        if base_actor_obs_dim <= 0 or base_actor_obs_dim >= num_actor_obs:
            raise ValueError(
                f"base_actor_obs_dim must be between 1 and {num_actor_obs - 1}, got {base_actor_obs_dim}."
            )
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive.")
        if not 0.0 <= active_base_action_scale <= 1.0:
            raise ValueError("active_base_action_scale must be between 0 and 1.")
        if initialize_stair_actor_from_blind and (
            list(residual_hidden_dims) != list(actor_hidden_dims)
            or squash_residual or residual_scale != 1.0 or active_base_action_scale != 0.0
            or (residual_action_scales is not None and any(scale != 1.0 for scale in residual_action_scales))
        ):
            raise ValueError("Blind-initialized stair actor requires matching hidden layers and unscaled linear output.")
        if residual_action_scales is not None and (
            len(residual_action_scales) != num_actions
            or any(scale <= 0.0 for scale in residual_action_scales)
        ):
            raise ValueError("residual_action_scales must have one positive value per action.")

        super().__init__(
            num_actor_obs=base_actor_obs_dim,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        residual_activation = resolve_nn_activation(activation)
        residual_layers = []
        residual_input_dim = num_actor_obs
        for hidden_dim in residual_hidden_dims:
            residual_layers.append(nn.Linear(residual_input_dim, hidden_dim))
            residual_layers.append(residual_activation)
            residual_input_dim = hidden_dim
        residual_layers.append(nn.Linear(residual_input_dim, num_actions))
        residual_actor = nn.Sequential(*residual_layers)
        nn.init.zeros_(residual_layers[-1].weight)
        nn.init.zeros_(residual_layers[-1].bias)

        base_actor = self.actor
        for parameter in base_actor.parameters():
            parameter.requires_grad_(False)
        self.actor = _GatedResidualActor(
            base_actor=base_actor,
            residual_actor=residual_actor,
            base_actor_obs_dim=base_actor_obs_dim,
            gate_obs_index=gate_obs_index,
            residual_scale=residual_scale,
            residual_action_scales=residual_action_scales,
            active_base_action_scale=active_base_action_scale,
            squash_residual=squash_residual,
            residual_observation_scales=residual_observation_scales,
            stopping_phase_index=base_actor_obs_dim+1 if train_stair_conditioning_only else -1,
        )
        if train_stair_conditioning_only:
            if not initialize_stair_actor_from_blind:
                raise ValueError("Conditioning-only training requires an independent full stair actor.")
            self.actor.residual_actor.requires_grad_(False)
            weight = self.actor.residual_actor[0].weight
            weight.requires_grad_(True)
            weight.register_hook(lambda gradient: torch.cat((
                torch.zeros_like(gradient[:, :base_actor_obs_dim]), gradient[:, base_actor_obs_dim:]), dim=1))
        self.base_actor_obs_dim = base_actor_obs_dim
        self.initialize_stair_actor_from_blind = initialize_stair_actor_from_blind
        self.min_action_std = min_action_std
        self.max_action_std = max_action_std
        self.low_noise_obs_indices = list(low_noise_obs_indices or [])
        self.low_noise_scale = low_noise_scale
        if not 0.0 < low_noise_scale <= 1.0 or any(
            index < 0 or index >= num_actor_obs for index in self.low_noise_obs_indices
        ):
            raise ValueError("Invalid phase-conditioned exploration configuration.")
        self.reset_optimizer_on_load = False
        print(f"Gated residual actor: {self.actor}")

    def update_distribution(self, observations):
        mean = self.actor(observations)
        if self.noise_std_type == "scalar":
            std = self.std.clamp(self.min_action_std, self.max_action_std).expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).clamp(self.min_action_std, self.max_action_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
        if self.low_noise_obs_indices:
            stopping = observations[:, self.low_noise_obs_indices].amax(dim=1, keepdim=True) > .5
            std = std * torch.where(stopping, self.low_noise_scale, 1.)
        self.distribution = Normal(mean, std)

    def load_state_dict(self, state_dict, strict=True):
        # A residual checkpoint can resume normally. The blind checkpoint uses
        # actor.* keys and is deliberately treated as initialization, not a
        # continuation of its learning-iteration counter or optimizer state.
        if any(key.startswith("actor.base_actor.") for key in state_dict):
            adapted = dict(state_dict)
            adapted.setdefault("actor.stopping_phase_index", self.actor.stopping_phase_index.clone())
            scale_key = "actor.residual_observation_scales"
            new_scale = self.actor.residual_observation_scales
            old_scale = adapted.get(scale_key, torch.ones_like(new_scale)).to(new_scale)
            self.reset_optimizer_on_load = not torch.equal(old_scale, new_scale)
            if self.reset_optimizer_on_load:
                # Change coordinates without changing the loaded action function.
                key = "actor.residual_actor.0.weight"
                adapted[key] = adapted[key]*(old_scale/new_scale).to(adapted[key])
            adapted[scale_key] = new_scale.clone()
            nn.Module.load_state_dict(self, adapted, strict=strict)
            return True

        loaded_actor_input = state_dict.get("actor.0.weight")
        self.reset_optimizer_on_load = False
        if loaded_actor_input is None:
            raise KeyError("Blind checkpoint does not contain actor.0.weight.")
        if loaded_actor_input.shape[1] != self.base_actor_obs_dim:
            raise ValueError(
                "Blind checkpoint actor observation size does not match the frozen base: "
                f"{loaded_actor_input.shape[1]} != {self.base_actor_obs_dim}."
            )

        adapted_state = dict(self.state_dict())
        copied_keys = []
        for key, value in state_dict.items():
            # New reward and stop/shift phases need their own critic and noise.
            # A complete stair checkpoint above still resumes both normally.
            if self.initialize_stair_actor_from_blind and (key.startswith("critic.") or key in ("std", "log_std")):
                continue
            if key.startswith("actor."):
                target_key = "actor.base_actor." + key[len("actor.") :]
            else:
                target_key = key
            if target_key not in adapted_state:
                continue
            target = adapted_state[target_key]
            if value.shape == target.shape:
                adapted_state[target_key] = value
                copied_keys.append(target_key)
            elif target_key == "critic.0.weight" and value.ndim == 2 and value.shape[0] == target.shape[0]:
                if value.shape[1] > target.shape[1]:
                    raise ValueError("Blind critic has more observations than the residual critic.")
                expanded = target.clone()
                expanded[:, : value.shape[1]] = value
                expanded[:, value.shape[1] :] = 0.0
                adapted_state[target_key] = expanded
                copied_keys.append(target_key)
            elif target_key.startswith("actor.base_actor.") or target_key.startswith("critic."):
                raise ValueError(
                    f"Cannot load blind checkpoint tensor {key}: {tuple(value.shape)} -> {tuple(target.shape)}."
                )

        if self.initialize_stair_actor_from_blind:
            for key, value in state_dict.items():
                if not key.startswith("actor."):
                    continue
                target_key = "actor.residual_actor."+key[len("actor."):]
                target = adapted_state[target_key]
                if key == "actor.0.weight":
                    expanded = torch.zeros_like(target)
                    expanded[:, :self.base_actor_obs_dim] = value
                    adapted_state[target_key] = expanded
                else:
                    adapted_state[target_key] = value.clone()
            print("[INFO] Initialized independent stair actor from blind actor; new feature weights are zero.")

        nn.Module.load_state_dict(self, adapted_state, strict=True)
        for parameter in self.actor.base_actor.parameters():
            parameter.requires_grad_(False)
        print(f"[INFO] Initialized frozen blind actor from {len(copied_keys)} checkpoint tensors.")
        return False
