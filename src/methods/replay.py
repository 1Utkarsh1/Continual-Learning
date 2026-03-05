from src.methods.baseline import BaselineLearner
import torch
import random

class ExperienceReplayLearner(BaselineLearner):
    def __init__(self, model, device, learning_rate, buffer_size, replay_batch_size):
        super().__init__(model, device, learning_rate)
        self.buffer_size = buffer_size
        self.replay_batch_size = replay_batch_size
        self.buffer = []

    def train(self, train_loader, val_loader=None, task_id=0, epochs=1, eval_freq=1, **kwargs):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # Incorporate replay loss if buffer is not empty
                if len(self.buffer) > 0:
                    replay_samples = random.sample(self.buffer, min(self.replay_batch_size, len(self.buffer)))
                    replay_inputs = torch.stack([sample[0] for sample in replay_samples]).to(self.device)
                    replay_targets = torch.tensor([sample[1] for sample in replay_samples]).to(self.device)
                    replay_outputs = self.model(replay_inputs)
                    replay_loss = criterion(replay_outputs, replay_targets)
                    loss = loss + replay_loss

                loss.backward()
                optimizer.step()

        # Update the replay buffer with current task samples
        for batch in train_loader:
            inputs, targets = batch
            for i in range(inputs.size(0)):
                self.buffer.append((inputs[i].detach().cpu(), targets[i].detach().cpu()))
        # Keep buffer size within limits
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:] 