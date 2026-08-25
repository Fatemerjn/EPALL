import torch.nn.functional as F
from .er import *


class Derpp(ER):
    def __init__(self, args):
        super(Derpp, self).__init__(args)
        self.memory = RehearsalMemory(buffer_size=self.args.mem_budget,
                                      n_tasks=self.args.n_tasks,
                                      cpt=self.args.class_per_task,
                                      dim_x=self.args.dim_input,
                                      device=self.device,
                                      mem_type=self.args.mem_type,
                                      save_logits=True)
        self.alpha = args.alpha
        self.beta = args.beta

    def der_loss(self, a, b, task_id):
        a_ = a[..., task_id*self.cpt:(task_id+1)*self.cpt]
        b_ = b[..., task_id*self.cpt:(task_id+1)*self.cpt]
        return F.mse_loss(a_, b_)

    def learn(self, task_id, dataset):
        loader = self.build_dataloader(dataset, shuffle=True, context=f"learn_task_{task_id}")
        self.opt = self.init_optimizer()
        train_start = time.perf_counter()
        self.log_progress(f"task training start: task={task_id} epochs={self.args.n_epochs} steps_per_epoch={len(loader)}")

        if isinstance(self.net, SubnetVisionTransformer) or isinstance(self.net, VisionTransformer):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args.n_epochs)

        self.n_iters = self.args.n_epochs * len(loader)
        for epoch in range(self.args.n_epochs):
            epoch_start = time.perf_counter()
            self.log_epoch_start(task_id, epoch, self.args.n_epochs)
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                h = self.forward(x, task_id)
                loss = self.loss_fn(h, y)

                n_prev_tasks = len(self.prev_tasks)
                for t in self.prev_tasks:
                    x_past, _, h_past = self.memory.sample_task(self.args.batch_size // n_prev_tasks, t)
                    loss += self.alpha * self.der_loss(self.forward(x_past, t), h_past, t) / n_prev_tasks
                    if self.beta > 0.0:
                        x_past, y_past, _ = self.memory.sample_task(self.args.batch_size // n_prev_tasks, t)
                        loss += self.beta * self.loss_fn(self.forward(x_past, t), y_past) / n_prev_tasks
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            if self.scheduler is not None:
                self.scheduler.step()
            self.log_epoch_end(task_id, epoch, self.args.n_epochs, epoch_start)

        self.fill_buffer(task_id, dataset)
        self.prev_tasks.append(task_id)
        self.log_progress(
            f"task training end: task={task_id} elapsed={self._format_elapsed(self._elapsed_since(train_start))}"
        )

    def forget(self, task_id):
        self.prev_tasks.remove(task_id)
        n_prev_tasks = len(self.prev_tasks)

        self.opt = self.init_optimizer()

        assert self.args.forget_iters is not None
        forget_start = time.perf_counter()
        self.log_progress(f"forget phase start: task={task_id} iterations={self.args.forget_iters}")

        for i in range(self.args.forget_iters):
            self.opt.zero_grad()
            x_forget, _, _ = self.memory.sample_task(self.args.batch_size, task_id)
            out = self.forward(x_forget, task_id)

            uniform_target = (torch.ones(out.shape) / self.cpt).to(self.device)
            if task_id > 0:
                uniform_target[:, :self.cpt * task_id].data.fill_(0.0)
            if task_id < self.n_tasks - 1:
                uniform_target[:, self.cpt * (task_id + 1):].data.fill_(0.0)

            loss = self.loss_fn(out, uniform_target)

            if n_prev_tasks > 0:
                for t in self.prev_tasks:
                    x_past, _, h_past = self.memory.sample_task(self.args.batch_size // n_prev_tasks, t)
                    loss += self.alpha * self.der_loss(self.forward(x_past, t), h_past, t) / n_prev_tasks
                    if self.beta > 0.0:
                        x_past, y_past, _ = self.memory.sample_task(self.args.batch_size // n_prev_tasks, t)
                        loss += self.beta * self.loss_fn(self.forward(x_past, t), y_past) / n_prev_tasks

            loss.backward()
            self.opt.step()

        self.memory.remove(task_id)
        self.log_progress(
            f"forget phase end: task={task_id} elapsed={self._format_elapsed(self._elapsed_since(forget_start))}"
        )
