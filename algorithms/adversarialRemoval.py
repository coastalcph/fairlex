import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from scheduler import initialize_scheduler
from optimizer import initialize_optimizer
from torch.nn.utils import clip_grad_norm_
from pytorch_revgrad import RevGrad


class AdversarialRemoval(SingleModelAlgorithm):
    """
        Adversarial Attribute Removal.

        Original paper:
            @inproceedings{elazar-goldberg-2018-adversarial,
            title = "Adversarial Removal of Demographic Attributes from Text Data",
                author = "Elazar, Yanai  and Goldberg, Yoav",
                booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
                year = "2018",
                address = "Brussels, Belgium",
                publisher = "Association for Computational Linguistics",
                url = "https://www.aclweb.org/anthology/D18-1002",
            }

        """
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):

        featurizer, classifier = initialize_model(config, d_out=d_out, is_featurizer=True)
        featurizer = featurizer.to(config.device)
        classifier = classifier.to(config.device)
        # extra modules for adversarial classifier
        adv_classifier = torch.nn.Linear(featurizer.d_out, grouper.n_groups if grouper.n_groups > 2 else 1)
        rev_grad = RevGrad(alpha=config.adv_lambda)
        adv_classifier = adv_classifier.to(config.device)
        rev_grad = rev_grad.to(config.device)
        self.adv_lambda = config.adv_lambda

        # set models
        model = torch.nn.Sequential(featurizer, classifier).to(config.device)

        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )

        # set model components
        self.featurizer = featurizer
        self.classifier = classifier
        self.adv_classifier = adv_classifier
        # self.logged_fields.append('adv_objective')

        # initialize adversarial model, optimizer, and scheduler
        self.adv_model = torch.nn.Sequential(featurizer, rev_grad, adv_classifier).to(config.device)
        self.module_list = torch.nn.ModuleList([featurizer, classifier, adv_classifier])
        self.optimizer = initialize_optimizer(config, self.module_list)
        self.scheduler = initialize_scheduler(config, self.optimizer, n_train_steps)
        self.schedulers = [self.scheduler]

    def process_batch(self, batch):
        """
        Override
        """
        # forward pass for both classifier and adversarial classifier
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        with torch.cuda.amp.autocast(enabled=self.use_scaler):
            features = self.featurizer(x)
            outputs = self.classifier(features)
            adv_outputs = self.adv_classifier(features)

        # package the results
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'adv_y_pred': adv_outputs,
            'metadata': metadata,
            }
        return results

    def objective(self, results):
        adv_preds = results['adv_y_pred']
        if self.grouper.n_groups > 2:
            adv_preds = torch.argmax(results['adv_y_pred'], -1, keepdim=True)
            
        avg_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
        adv_loss = self.loss.compute(adv_preds, results['g'].unsqueeze(1), return_dict=False)

        return avg_loss,  adv_loss

    def _update(self, results):
        """
        Computes the objective and updates the model.
        Also updates the results dictionary yielded by process_batch().
        Should be overridden to change algorithm update beyond modifying the objective.
        """
        # compute objective
        with torch.cuda.amp.autocast(enabled=self.use_scaler):
            objective = self.objective(results)
            # combine losses
            total_objective = objective[0] + objective[1]
        results['objective'] = objective[0].item()
        results['adv_objective'] = objective[1].item()
        self.model.zero_grad()
        self.adv_model.zero_grad()
        # update
        if self.use_scaler:
            self.scaler.scale(total_objective).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            total_objective.backward()
        # update
        if self.max_grad_norm:
            clip_grad_norm_(list(self.model.parameters()) + list(self.adv_model.parameters()), self.max_grad_norm)
        if self.use_scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.step_schedulers(
            is_epoch=False,
            metrics=results,
            log_access=False)
