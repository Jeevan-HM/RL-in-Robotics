from dataclasses import dataclass
import numpy as np

@dataclass
class Transition:
    obs: np.ndarray
    act: np.ndarray
    rew: float
    done: bool
    next_obs: np.ndarray

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size:int):
        self.obs_buf  = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf  = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf  = np.zeros((size, ), dtype=np.float32)
        self.done_buf = np.zeros((size, ), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, done, next_obs):
        self.obs_buf[self.ptr]  = obs
        self.act_buf[self.ptr]  = act
        self.rew_buf[self.ptr]  = rew
        self.done_buf[self.ptr] = done
        self.obs2_buf[self.ptr] = next_obs
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     obs2=self.obs2_buf[idxs])
        return batch
