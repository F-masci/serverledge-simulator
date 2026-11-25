"""Modulo PyTorch Ray RLlib per PPO con action masking (New API Stack)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gymnasium import spaces
import torch
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import FLOAT_MIN


class ActionMaskingPPOTorchRLModule(DefaultPPOTorchRLModule):
    """
    Modulo PyTorch RLlib per PPO con action masking su spazio osservazione Dict.

    Si aspetta osservazioni con chiavi:
      - 'observations': Box(shape=(obs_dim,), dtype=float32)
      - 'action_mask': Box(0.0, 1.0, shape=(action_dim,), dtype=float32)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        obs_space = kwargs.get('observation_space')
        if obs_space is None and args:
            obs_space = args[0]
        if not isinstance(obs_space, spaces.Dict):
            raise ValueError(
                'ActionMaskingPPOTorchRLModule richiede uno spazio osservazione gymnasium.spaces.Dict'
            )
        if 'observations' not in obs_space.spaces or 'action_mask' not in obs_space.spaces:
            raise ValueError("Lo spazio osservazione deve contenere le chiavi 'observations' e 'action_mask'")

        # Passa al catalogo RLlib il solo spazio 'observations', poi ripristina il Dict in setup().
        self._full_observation_space = obs_space
        if 'observation_space' in kwargs:
            kwargs['observation_space'] = obs_space['observations']
        else:
            args = (obs_space['observations'], *args[1:])
        super().__init__(*args, **kwargs)

    @override(DefaultPPOTorchRLModule)
    def setup(self) -> None:
        super().setup()
        self.observation_space = self._full_observation_space

    @staticmethod
    def _split_batch(batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        obs = batch[Columns.OBS]
        if not isinstance(obs, Mapping) or 'observations' not in obs or 'action_mask' not in obs:
            raise ValueError("Il batch deve contenere obs['observations'] e obs['action_mask']")
        action_mask = obs['action_mask']
        inner_batch = {**batch, Columns.OBS: obs['observations']}
        return action_mask, inner_batch

    @staticmethod
    def _mask_logits(outs: dict[str, Any], action_mask: torch.Tensor) -> dict[str, Any]:
        logits = outs[Columns.ACTION_DIST_INPUTS]
        mask = action_mask.to(device=logits.device) > 0.5
        outs[Columns.ACTION_DIST_INPUTS] = torch.where(
            mask, logits, torch.full_like(logits, FLOAT_MIN),
        )
        return outs

    @override(DefaultPPOTorchRLModule)
    def _forward_inference(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        action_mask, inner = self._split_batch(batch)
        outs = super()._forward_inference(inner, **kwargs)
        return self._mask_logits(outs, action_mask)

    @override(DefaultPPOTorchRLModule)
    def _forward_exploration(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        action_mask, inner = self._split_batch(batch)
        outs = super()._forward_exploration(inner, **kwargs)
        return self._mask_logits(outs, action_mask)

    @override(DefaultPPOTorchRLModule)
    def _forward_train(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        action_mask, inner = self._split_batch(batch)
        outs = super()._forward_train(inner, **kwargs)
        return self._mask_logits(outs, action_mask)

    @override(DefaultPPOTorchRLModule)
    def compute_values(self, batch: dict[str, Any], embeddings: Any = None) -> torch.Tensor:
        if isinstance(batch[Columns.OBS], Mapping):
            _, inner = self._split_batch(batch)
        else:
            inner = batch
        return super().compute_values(inner, embeddings)
