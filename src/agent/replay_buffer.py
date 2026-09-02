from collections import deque
import random

import numpy as np
import torch


class ReplayBuffer:
    """
    Stores transitions:

        (state, action, reward, next_state, done)

    and allows random mini-batch sampling.
    """

    def __init__(
        self,
        capacity=50_000,
    ):

        self.buffer = deque(
            maxlen=capacity
        )


    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        transition = (
            state,
            action,
            reward,
            next_state,
            done,
        )

        self.buffer.append(
            transition
        )


    def sample(
        self,
        batch_size,
        device="cpu",
    ):

        batch = random.sample(
            self.buffer,
            batch_size,
        )

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = zip(*batch)


        states = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device=device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=device,
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=device,
        )

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=device,
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
        )


    def __len__(self):
        return len(self.buffer)
