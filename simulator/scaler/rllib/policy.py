"""Wrapper di inferenza PyTorch RLlib e caricamento checkpoint (senza cluster Ray)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec

from .module import ActionMaskingPPOTorchRLModule
from ..config import PPOConfig
from ..contracts import ScalingPolicy


def load_rllib_module(checkpoint_path: str | Path) -> RLModule:
    """Carica un RLModule da un checkpoint di algoritmo RLlib o direttamente dalla directory modulo."""
    path = Path(checkpoint_path).resolve()
    nested = path / 'learner_group' / 'learner' / 'rl_module' / 'default_policy'
    target = nested if nested.exists() else path
    return RLModule.from_checkpoint(target)


class RLlibPPOPolicy(ScalingPolicy):
    """
    Wrapper leggero di inferenza sul modulo PyTorch `ActionMaskingPPOTorchRLModule` di RLlib.

    Consente l'esecuzione della policy sia nel simulatore sia nel servizio gRPC
    dello Zone Controller in `serverledge-infrascaler` senza avviare un cluster Ray.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: PPOConfig,
        *,
        module: RLModule | None = None,
        seed: int | None = None,
    ) -> None:
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._config = config

        if module is not None:
            self._module = module
        elif config.checkpoint_path is not None:
            self._module = load_rllib_module(config.checkpoint_path)
        else:
            if seed is not None:
                torch.manual_seed(seed)
            obs_space = spaces.Dict({
                'observations': spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
                'action_mask': spaces.Box(low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32),
            })
            act_space = spaces.Discrete(action_dim)
            spec = RLModuleSpec(
                module_class=ActionMaskingPPOTorchRLModule,
                observation_space=obs_space,
                action_space=act_space,
                model_config=DefaultModelConfig(
                    fcnet_hiddens=list(config.hidden_sizes),
                    fcnet_activation='tanh',
                ),
            )
            self._module = spec.build()

        self._module.eval()

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def module(self) -> RLModule:
        return self._module

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        *,
        rng: np.random.Generator,
        deterministic: bool | None = None,
    ) -> int:
        """Seleziona un'azione ammissibile tramite forward_inference del modulo RLlib."""
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        mask_arr = np.asarray(action_mask, dtype=bool)
        if not np.any(mask_arr):
            raise ValueError('Nessuna azione ammissibile nella maschera')
        mask_t = torch.as_tensor(mask_arr.astype(np.float32)).unsqueeze(0)

        with torch.no_grad():
            outs = self._module.forward_inference({
                Columns.OBS: {'observations': obs_t, 'action_mask': mask_t},
            })
            logits = outs[Columns.ACTION_DIST_INPUTS][0].detach().cpu().numpy()

        use_det = self._config.deterministic if deterministic is None else deterministic
        masked_logits = np.where(mask_arr, logits, -np.inf)
        if use_det:
            return int(np.argmax(masked_logits))

        shifted = np.where(mask_arr, masked_logits - np.max(masked_logits), -np.inf)
        probs = np.where(mask_arr, np.exp(shifted), 0.0)
        probs = probs / np.sum(probs)
        return int(rng.choice(self._action_dim, p=probs))

    def save(self, path: str | Path) -> Path:
        """Salva il modulo RLlib su disco (formato nativo RLlib checkpoint)."""
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self._module.save_to_path(target)
        return target
