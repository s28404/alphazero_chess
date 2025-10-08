import wandb
import torch
import os

class Trainer:
    def __init__(
        self,
        model,
        env,
        seed,
        optimizer,
        scheduler,
        scheduler_interval,
        loss_fn,
        checkpoint_interval,
        device,
        replay_buffer,
        batch_size=64,
        train_steps_per_iteration=1000,
    ):
        self.model = model
        self.env = env
        self.seed = seed
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.loss_fn = loss_fn
        self.checkpoint_interval = checkpoint_interval
        self.device = device
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.train_steps_per_iteration = train_steps_per_iteration
        self.iteration = 0
        self.total_steps = 0

    def train_step(self, batch_states, batch_policies, batch_values):
        """
        Perform a single gradient update on a batch of data.

        Args:
            batch_states: Batch of board states
            batch_policies: Batch of target policy distributions
            batch_values: Batch of target values

        Returns:
            Dictionary containing loss statistics for this step
        """
        self.model.train()

        # Move data to device
        batch_states = batch_states.to(self.device)
        batch_policies = batch_policies.to(self.device)
        batch_values = batch_values.to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        pred_policy, pred_value = self.model(batch_states)

        # Compute losses
        value_loss = self.loss_fn(pred_value.view(-1), batch_values)
        policy_loss = self.loss_fn(pred_policy, batch_policies)
        loss = value_loss + policy_loss

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Return statistics
        return {
            "step_loss": loss.item(),
            "step_policy_loss": policy_loss.item(),
            "step_value_loss": value_loss.item(),
        }

    def sample_batch(self):
        """
        Sample a batch from the replay buffer.

        Returns:
            Tuple of (batch_states, batch_policies, batch_values) tensors
        """
        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.stack([item[0] for item in batch])
        policies = torch.stack(
            [torch.tensor(item[1], dtype=torch.float32) for item in batch]
        )
        values = torch.tensor([item[2] for item in batch], dtype=torch.float32)

        return states, policies, values

    def train_episode(self, episode_data):
        """
        Train on data from self-play episodes.
        Performs multiple training steps by sampling from replay buffer.

        Args:
            episode_data: List of (state, policy, value) tuples from one or more self-play episodes

        Returns:
            Dictionary containing average training statistics
        """
        # Add episode data to replay buffer
        for state, policy, value in episode_data:
            self.replay_buffer.add(state, policy, value)

        # Check if we have enough data to train
        if len(self.replay_buffer) < self.batch_size:
            print(f"Not enough data in replay buffer (need at least {self.batch_size})")
            return None

        # Perform training steps
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for step in range(self.train_steps_per_iteration):
            # Sample a batch from replay buffer
            batch_states, batch_policies, batch_values = self.sample_batch()

            # Perform one gradient update
            step_stats = self.train_step(batch_states, batch_policies, batch_values)

            total_loss += step_stats["step_loss"]
            total_policy_loss += step_stats["step_policy_loss"]
            total_value_loss += step_stats["step_value_loss"]
            self.total_steps += 1

            if (step + 1) % 100 == 0:
                avg_loss = total_loss / (step + 1)
                avg_policy_loss = total_policy_loss / (step + 1)
                avg_value_loss = total_value_loss / (step + 1)
                print(
                    f"  Step {step + 1}/{self.train_steps_per_iteration} - "
                    f"Loss: {avg_loss:.4f}, "
                    f"Policy: {avg_policy_loss:.4f}, "
                    f"Value: {avg_value_loss:.4f}"
                )

        # Calculate and return average statistics
        return {
            "avg_loss": total_loss / self.train_steps_per_iteration,
            "avg_policy_loss": total_policy_loss / self.train_steps_per_iteration,
            "avg_value_loss": total_value_loss / self.train_steps_per_iteration,
        }

    def train(self, training_data):
        """
        Main training loop for one iteration.
        Orchestrates the training process: adds episodes to buffer and trains the model.

        Args:
            training_data: List of (state, policy, value) tuples from self-play episodes
        """
        if len(training_data) == 0:
            print("No training data available")
            return

        print(f"Training on {len(training_data)} samples from self-play")
        print(f"Replay buffer size before: {len(self.replay_buffer)}")

        # Train on the episode data
        episode_stats = self.train_episode(training_data)

        if episode_stats is None:
            return

        print(f"Replay buffer size after: {len(self.replay_buffer)}")
        print(
            f"Training Complete - Avg Loss: {episode_stats['avg_loss']:.4f}, "
            f"Policy Loss: {episode_stats['avg_policy_loss']:.4f}, "
            f"Value Loss: {episode_stats['avg_value_loss']:.4f}"
        )

        # Log to wandb
        wandb.log(
            {
                "iteration": self.iteration,
                "total_steps": self.total_steps,
                "avg_loss": episode_stats["avg_loss"],
                "avg_policy_loss": episode_stats["avg_policy_loss"],
                "avg_value_loss": episode_stats["avg_value_loss"],
                "replay_buffer_size": len(self.replay_buffer),
            }
        )

        # Update learning rate scheduler
        self.iteration += 1
        if self.iteration % self.scheduler_interval == 0:
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"Learning rate adjusted to: {current_lr:.6f}")
            wandb.log({"learning_rate": current_lr})

        if self.iteration % self.checkpoint_interval == 0:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = f"checkpoints/model_iter_{self.iteration}_seed_{self.seed}.pth"
            torch.save({
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            }, checkpoint_path)
        
        return episode_stats
