from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .agent import SACAgent

__all__ = ["save_agent", "load_agent"]


def _ensure_checkpoint_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _to_serializable_meta(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return value


def save_agent(agent: SACAgent, path: str | Path, metadata: Optional[Dict[str, Any]] = None):
    meta = {k: _to_serializable_meta(v) for k, v in (metadata or {}).items()}
    checkpoint = {
        "actor": agent.actor.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "q1_targ": agent.q1_targ.state_dict(),
        "q2_targ": agent.q2_targ.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "actor_opt": agent.actor_opt.state_dict(),
        "q1_opt": agent.q1_opt.state_dict(),
        "q2_opt": agent.q2_opt.state_dict(),
        "alpha_opt": agent.alpha_opt.state_dict(),
        "metadata": meta,
    }
    torch.save(checkpoint, _ensure_checkpoint_path(path))


def load_agent(agent: SACAgent, path: str | Path, map_location: Optional[str] = None) -> Optional[Dict[str, Any]]:
    load_kwargs = {"map_location": map_location or agent.device, "weights_only": False}
    try:
        checkpoint = torch.load(Path(path), **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        checkpoint = torch.load(Path(path), **load_kwargs)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.q1.load_state_dict(checkpoint["q1"])
    agent.q2.load_state_dict(checkpoint["q2"])
    agent.q1_targ.load_state_dict(checkpoint["q1_targ"])
    agent.q2_targ.load_state_dict(checkpoint["q2_targ"])
    log_alpha = checkpoint.get("log_alpha")
    if log_alpha is not None:
        if isinstance(log_alpha, torch.Tensor):
            agent.log_alpha.data = log_alpha.to(agent.device)
        else:
            agent.log_alpha.data = torch.tensor(float(log_alpha), device=agent.device)
    if "actor_opt" in checkpoint:
        agent.actor_opt.load_state_dict(checkpoint["actor_opt"])
    if "q1_opt" in checkpoint:
        agent.q1_opt.load_state_dict(checkpoint["q1_opt"])
    if "q2_opt" in checkpoint:
        agent.q2_opt.load_state_dict(checkpoint["q2_opt"])
    if agent.auto_alpha and "alpha_opt" in checkpoint:
        agent.alpha_opt.load_state_dict(checkpoint["alpha_opt"])
    return checkpoint.get("metadata")
