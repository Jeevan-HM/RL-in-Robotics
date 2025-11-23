from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .agent import Stage2Agent

__all__ = ["save_stage2_agent", "load_stage2_agent"]


def save_stage2_agent(agent: Stage2Agent, path: str | Path, metadata: Optional[dict] = None):
    checkpoint = {
        "actor": agent.actor.state_dict(),
        "safety_q": agent.safety_q.state_dict(),
        "nav_q": agent.nav_q.state_dict(),
        "safety_q_targ": agent.safety_q_targ.state_dict(),
        "nav_q_targ": agent.nav_q_targ.state_dict(),
        "actor_opt": agent.actor_opt.state_dict(),
        "safety_q_opt": agent.safety_q_opt.state_dict(),
        "nav_q_opt": agent.nav_q_opt.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(checkpoint, Path(path))


def load_stage2_agent(agent: Stage2Agent, path: str | Path, map_location: Optional[str] = None) -> Optional[dict]:
    load_kwargs = {"map_location": map_location or agent.device, "weights_only": False}
    try:
        checkpoint = torch.load(Path(path), **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        checkpoint = torch.load(Path(path), **load_kwargs)

    agent.actor.load_state_dict(checkpoint["actor"])
    agent.safety_q.load_state_dict(checkpoint["safety_q"])
    agent.nav_q.load_state_dict(checkpoint["nav_q"])
    if "safety_q_targ" in checkpoint:
        agent.safety_q_targ.load_state_dict(checkpoint["safety_q_targ"])
    else:
        agent.safety_q_targ.load_state_dict(agent.safety_q.state_dict())
    if "nav_q_targ" in checkpoint:
        agent.nav_q_targ.load_state_dict(checkpoint["nav_q_targ"])
    else:
        agent.nav_q_targ.load_state_dict(agent.nav_q.state_dict())

    if "actor_opt" in checkpoint:
        agent.actor_opt.load_state_dict(checkpoint["actor_opt"])
    if "safety_q_opt" in checkpoint:
        agent.safety_q_opt.load_state_dict(checkpoint["safety_q_opt"])
    if "nav_q_opt" in checkpoint:
        agent.nav_q_opt.load_state_dict(checkpoint["nav_q_opt"])

    return checkpoint.get("metadata")
