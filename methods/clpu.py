import copy
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from .base import *
import models
from models.vit import VisionTransformer
from models.subnet_vit import SubnetVisionTransformer


class CLPU(Base):
    def __init__(self, args):
        super(CLPU, self).__init__(args)
        self.side_nets = {}

    def eval_mode(self):
        self.eval()
        for t in self.side_nets:
            self.side_nets[t].eval()

    def train_mode(self):
        self.train()
        for t in self.side_nets:
            self.side_nets[t].train()

    def init_optimizer_per_task(self, task_id):
        if self.args.optim == "sgd":
            return SGD(self.side_nets[task_id].parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.args.optim == "adam":
            return Adam(self.side_nets[task_id].parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

    def forward(self, x, task):
        if task in self.side_nets:
            pred = self._forward_net(self.side_nets[task], x, task)
        else:
            pred = self._forward_net(self.net, x, task)
        if task > 0:
            pred[:, :self.cpt*task].data.fill_(-10e10)
        if task < self.n_tasks-1:
            pred[:, self.cpt*(task+1):].data.fill_(-10e10)
        return pred

    def forward_with_features(self, x, task):
        if task in self.side_nets:
            pred, features = self._forward_net(self.side_nets[task], x, task, returnt='all')
        else:
            pred, features = self._forward_net(self.net, x, task, returnt='all')
        if task > 0:
            pred[:, :self.cpt * task].data.fill_(-10e10)
        if task < self.n_tasks - 1:
            pred[:, self.cpt * (task + 1):].data.fill_(-10e10)
        return pred, features

    def learn(self, task_id, dataset):
        assert task_id not in self.side_nets, f"[ERROR] should not see {task_id} in side nets"

        self.side_nets[task_id] = models.__dict__[self.args.arch.lower()](self.cpt * self.n_tasks).to(self.args.device)

        loader = self.build_dataloader(dataset, shuffle=True, context=f"learn_task_{task_id}")
        opt = self.init_optimizer_per_task(task_id)
        train_start = time.perf_counter()
        self.log_progress(f"task training start: task={task_id} epochs={self.args.n_epochs} steps_per_epoch={len(loader)}")

        if isinstance(self.net, SubnetVisionTransformer) or isinstance(self.net, VisionTransformer):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.args.n_epochs)

        for epoch in range(self.args.n_epochs):
            epoch_start = time.perf_counter()
            self.log_epoch_start(task_id, epoch, self.args.n_epochs)
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                h = self.forward(x, task_id)
                loss = self.loss_fn(h, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if self.scheduler is not None:
                self.scheduler.step()
            self.log_epoch_end(task_id, epoch, self.args.n_epochs, epoch_start)
        self.log_progress(
            f"task training end: task={task_id} elapsed={self._format_elapsed(self._elapsed_since(train_start))}"
        )

    def forget(self, task_id):
        del self.side_nets[task_id]
