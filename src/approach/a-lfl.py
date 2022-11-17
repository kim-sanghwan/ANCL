import torch
from copy import deepcopy
from argparse import ArgumentParser

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing A-LFL approach in our paper"""
    # In original paper, they used lamb_a = 1.6e-3 & 3.9e-4 for MNIST and CIFAR10 dataset
    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, lamb=400, lamb_a=100):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.model_aux = None
        self.lamb = lamb
        self.lamb_a = lamb_a

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--lamb', default=400, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        parser.add_argument('--lamb-a', default=100, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
        print("Lamb: ", self.lamb)
        print("Lamb_a: ", self.lamb_a)
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
            self.model_aux.eval()
            self.model_aux.freeze_all()

        print('=' * 108)
        print("Training of Main Network")
        print('=' * 108)
        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Forward old model and auxiliary model
            targets_old = None
            features_old = None
            targets_aux = None
            features_aux = None
            if t > 0:
                targets_old, features_old = self.model_old(images.to(self.device), return_features=True)
                targets_aux, features_aux = self.model_aux(images.to(self.device), return_features=True)
            # Forward current model
            outputs, features = self.model(images.to(self.device), return_features=True)
            loss = self.criterion(t, outputs, targets.to(self.device), features, features_old, features_aux)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model and auxiliary model
                targets_old = None
                features_old = None
                targets_aux = None
                features_aux = None
                if t > 0:
                    targets_old, features_old = self.model_old(images.to(self.device), return_features=True)
                    targets_aux, features_aux = self.model_aux(images.to(self.device), return_features=True)
                # Forward current model
                outputs, features = self.model(images.to(self.device), return_features=True)
                loss = self.criterion(t, outputs, targets.to(self.device), features, features_old, features_aux)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, outputs, targets, features=None, features_old=None, features_aux=None):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            # Euclidean loss between current feature vector and previous feature vector
            # and Euclidean loss between current feature vector and new feature vector
            features = features - features.mean(dim=1, keepdim=True)
            features_old = features_old - features_old.mean(dim=1, keepdim=True)
            features_aux = features_aux - features_aux.mean(dim=1, keepdim=True)

            features = torch.nn.functional.normalize(features, p=2, dim=1)
            features_old = torch.nn.functional.normalize(features_old, p=2, dim=1)
            features_aux = torch.nn.functional.normalize(features_aux, p=2, dim=1)
            
            loss += self.lamb * torch.nn.functional.mse_loss(features, features_old)
            loss += self.lamb_a * torch.nn.functional.mse_loss(features, features_aux)
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