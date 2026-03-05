from src.methods.baseline import BaselineLearner
import torch
import torch.nn.functional as F
import copy

class LwFLearner(BaselineLearner):
    def __init__(self, model, device, learning_rate, temperature, alpha):
        super().__init__(model, device, learning_rate)
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_model = None  # Will hold a copy of the model from previous tasks

    def train(self, train_loader, val_loader=None, task_id=0, epochs=1, eval_freq=1, **kwargs):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion_ce = torch.nn.CrossEntropyLoss()
        criterion_kd = torch.nn.KLDivLoss(reduction='batchmean')
        T = self.temperature

        for epoch in range(epochs):
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss_ce = criterion_ce(outputs, targets)
                loss_kd = 0.0
                if self.teacher_model is not None:
                    self.teacher_model.eval()
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(inputs)
                    # Compute soft targets: student uses log_softmax, teacher uses softmax
                    soft_student = F.log_softmax(outputs / T, dim=1)
                    soft_teacher = F.softmax(teacher_outputs / T, dim=1)
                    loss_kd = criterion_kd(soft_student, soft_teacher) * (T * T)
                loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kd
                loss.backward()
                optimizer.step()
        
        # Update teacher model after training the current task
        self.teacher_model = copy.deepcopy(self.model) 