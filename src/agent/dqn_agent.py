import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from agent.network import QNetwork
from agent.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=64,
        replay_capacity=50_000,
        target_update_frequency=1000,
        device="cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.batch_size = batch_size

        self.target_update_frequency = (
            target_update_frequency
        )

        self.device = torch.device(device)

        # ==========================================
        # ONLINE NETWORK
        # ==========================================

        self.online_network = QNetwork(
            state_size=state_size,
            action_size=action_size,
        ).to(self.device)

        # ==========================================
        # TARGET NETWORK
        # ==========================================

        self.target_network = QNetwork(
            state_size=state_size,
            action_size=action_size,
        ).to(self.device)

        # Initially they must be identical.
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        self.target_network.eval()

        # ==========================================
        # OPTIMIZER
        # ==========================================

        self.optimizer = Adam(
            self.online_network.parameters(),
            lr=learning_rate,
        )

        # Huber loss.
        self.loss_function = nn.SmoothL1Loss()

        # ==========================================
        # REPLAY BUFFER
        # ==========================================

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity
        )

        self.training_steps = 0

    # ==========================================================
    # ACTION SELECTION
    # ==========================================================

    def select_action(
        self,
        state,
        epsilon=0.0,
    ):
        """
        Epsilon-greedy policy.

        epsilon = 1:
            completely random

        epsilon = 0:
            completely greedy
        """

        if random.random() < epsilon:
            return random.randrange(
                self.action_size
            )

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_network(
                state_tensor
            )

        action = torch.argmax(
            q_values,
            dim=1,
        ).item()

        return action

    # ==========================================================
    # STORE EXPERIENCE
    # ==========================================================

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
    ):
        self.replay_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=terminated,
        )

    # ==========================================================
    # LEARN
    # ==========================================================

    def learn(self):

        if len(self.replay_buffer) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            terminated,
        ) = self.replay_buffer.sample(
            batch_size=self.batch_size,
            device=self.device,
        )

        # ==========================================
        # CURRENT Q(s, a)
        # ==========================================

        all_q_values = self.online_network(
            states
        )

        current_q_values = (
            all_q_values
            .gather(
                1,
                actions.unsqueeze(1),
            )
            .squeeze(1)
        )

        # ==========================================
        # BELLMAN TARGET
        # ==========================================

        with torch.no_grad():

            next_q_values = (
                self.target_network(
                    next_states
                )
            )

            best_next_q_values = (
                next_q_values
                .max(dim=1)
                .values
            )

            targets = (
                rewards
                + self.gamma
                * best_next_q_values
                * (1.0 - terminated)
            )

        # ==========================================
        # LOSS
        # ==========================================

        loss = self.loss_function(
            current_q_values,
            targets,
        )

        # ==========================================
        # BACKPROP
        # ==========================================

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        self.training_steps += 1

        # ==========================================
        # TARGET NETWORK UPDATE
        # ==========================================

        if (
            self.training_steps
            % self.target_update_frequency
            == 0
        ):
            self.update_target_network()

        return loss.item()

    # ==========================================================
    # TARGET NETWORK
    # ==========================================================

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

