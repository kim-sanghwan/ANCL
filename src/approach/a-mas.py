import torch
import itertools
from argparse import ArgumentParser
from copy import deepcopy

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing A-MAS approach in our paper"""
    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, lamb=50, lamb_a =5, alpha=0.5, fi_num_samples=-1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.lamb = lamb
        self.lamb_a = lamb_a
        self.alpha = alpha
        self.num_samples = fi_num_samples
        self.model_aux = None
        self.optimizer_expand = None

        # In all cases, we only keep importance weights for the model, but not for the heads.
        feat_ext = self.model.model
        # Store current parameters as the initial parameters before first task starts
        self.older_params = {n: p.clone().detach() for n, p in feat_ext.named_parameters() if p.requires_grad}
        # Store importance
        self.importance = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext.named_parameters()
                           if p.requires_grad}
        # Parameter of auxiliary network
        self.auxiliary_params = None
        # Store importance for auxiliary network
        self.importance_aux = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext.named_parameters()
                       if p.requires_grad}

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # lambda is the regularizer trade-off -- In original code: MAS.ipynb block [4]: lambda set to 1
        parser.add_argument('--lamb', default=50, type=float, required=False,
                            help='Forgetting-intransigence trade-off  (default=%(default)s)')
        # lambda_e sets how important the new task is compared to the old one
        parser.add_argument('--lamb-a', default=5, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Define how old and new importance is fused, by default it is a 50-50 fusion
        parser.add_argument('--alpha', default=0.5, type=float, required=False,
                            help='A-MAS alpha (default=%(default)s)')
        # Number of samples from train for estimating importance
        parser.add_argument('--fi-num-samples', default=-1, type=int, required=False,
                            help='Number of samples for MAS Importance (-1: all available) (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    #= MAS (global) is implemented since the paper shows is more efficient than l-MAS (local)
    def estimate_parameter_importance(self, model, trn_loader):
        # Initialize importance matrices
        importance = {n: torch.zeros(p.shape).to(self.device) for n, p in model.model.named_parameters()
                      if p.requires_grad}
        # Compute fisher information for specified number of samples -- rounded to the batch size
        n_samples_batches = (self.num_samples // trn_loader.batch_size + 1) if self.num_samples > 0 \
            else (len(trn_loader.dataset) // trn_loader.batch_size)
        # Do forward and backward pass to accumulate L2-loss gradients
        model.train()
        for images, targets in itertools.islice(trn_loader, n_samples_batches):
            model.zero_grad()
            # MAS allows any unlabeled data to do the estimation, we choose the current data as in main experiments
            outputs = model.forward(images.to(self.device))
            # labels not required, "...use the gradients of the squared L2-norm of the learned function output."
            loss = torch.norm(torch.cat(outputs, dim=1), p=2, dim=1).mean()
            loss.backward()
            # accumulate the gradients over the inputs to obtain importance weights
            for n, p in model.model.named_parameters():
                if p.grad is not None:
                    importance[n] += p.grad.abs() * len(targets)
        # divide by N total number of samples
        n_samples = n_samples_batches * trn_loader.batch_size
        importance = {n: (p / n_samples) for n, p in importance.items()}
        return importance

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
        print("lamb : ", self.lamb)
        print("lamb_a : ", self.lamb_a)
        if t > 0:
            print('=' * 108)
            print("Training of Auxiliary Network")
            print('=' * 108)
            # Args for the new trainer
            new_trainer_args = dict(nepochs=self.nepochs, lr=self.lr, lr_min=self.lr_min, lr_factor=self.lr_factor,
                            lr_patience=self.lr_patience, clipgrad=self.clipgrad, momentum=0.9,
                            wd=5e-4, multi_softmax=self.multi_softmax, wu_nepochs=self.warmup_epochs,
                            wu_lr_factor=self.warmup_lr, fix_bn=self.fix_bn, logger=self.logger)
            self.model_aux = deepcopy(self.model)
            # Train auxiliary model on current dataset
            new_trainer = NewTaskTrainer(self.model_aux, self.device, **new_trainer_args)
            new_trainer.train_loop(t, trn_loader, val_loader)

            # Store parameter of auxiliary model to compute regularizer later
            self.auxiliary_params = {n: p.clone().detach() for n, p in self.model_aux.model.named_parameters() if p.requires_grad}

            # calculate importance of auxiliary model
            curr_importance = self.estimate_parameter_importance(self.model_aux, trn_loader)
            for n in self.importance_aux.keys():
                self.importance_aux[n] = curr_importance[n]

        print('=' * 108)
        print("Training of Main Network")
        print('=' * 108)
        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        # Store current parameters for the next task
        self.older_params = {n: p.clone().detach() for n, p in self.model.model.named_parameters() if p.requires_grad}

        # calculate importance
        curr_importance = self.estimate_parameter_importance(self.model, trn_loader)
        # merge importance, we do not want to keep importance for each task in memory
        for n in self.importance.keys():
            # Added option to accumulate importance over time with a pre-fixed growing alpha
            if self.alpha == -1:
                alpha = (sum(self.model.task_cls[:t]) / sum(self.model.task_cls)).to(self.device)
                self.importance[n] = alpha * self.importance[n] + (1 - alpha) * curr_importance[n]
            else:
                # As in original code: MAS_utils/MAS_based_Training.py line 638 -- just add prev and new
                self.importance[n] = self.alpha * self.importance[n] + (1 - self.alpha) * curr_importance[n]

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            loss_reg = 0
            # memory aware synapses regularizer penalty
            for n, p in self.model.model.named_parameters():
                if n in self.importance.keys():
                    loss_reg += torch.sum(self.importance[n] * (p - self.older_params[n]).pow(2)) / 2
            loss_reg_exp = 0
            # auxiliary network memory aware synapses regularizer penalty
            for n, p in self.model.model.named_parameters():
                if n in self.importance_aux.keys():
                    loss_reg_exp += torch.sum(self.importance_aux[n] * (p - self.auxiliary_params[n]).pow(2)) / 2            
            loss += self.lamb * loss_reg + self.lamb_a * loss_reg_exp
        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

class NewTaskTrainer(Inc_Learning_Appr):
    def __init__(self, model, device, nepochs=160, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=5e-4, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None):
        super(NewTaskTrainer, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad,
                                             momentum, wd, multi_softmax, wu_nepochs, wu_lr_factor, fix_bn,
                                             eval_on_train, logger)
