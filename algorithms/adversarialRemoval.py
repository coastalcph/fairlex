import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from scheduler import initialize_scheduler
from optimizer import initialize_optimizer
from torch.nn.utils import clip_grad_norm_
from pytorch_revgrad import RevGrad


class AdversarialRemoval(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):

        featurizer, classifier = initialize_model(config, d_out=d_out, is_featurizer=True)
        adv_classifier = torch.nn.Linear(featurizer.d_out, grouper.n_groups if grouper.n_groups > 2 else 1)
        rev_grad = RevGrad(alpha=config.adv_lambda)
        featurizer = featurizer.to(config.device)
        classifier = classifier.to(config.device)
        adv_classifier = adv_classifier.to(config.device)
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
        self.logged_fields.append('adv_objective')

        # initialize adversarial model, optimizer, and scheduler
        self.adv_model = torch.nn.Sequential(featurizer, rev_grad, adv_classifier).to(config.device)
        self.adv_optimizer = initialize_optimizer(config, self.adv_model)
        self.adv_scheduler = initialize_scheduler(config, self.optimizer, n_train_steps)
        self.schedulers.append(self.adv_scheduler)

    def process_batch(self, batch):
        """
        Override
        """
        # forward pass
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        if self.scaler:
            with torch.cuda.amp.autocast():
                features = self.featurizer(x)
                outputs = self.classifier(features)
                adv_outputs = self.adv_classifier(features)
        else:
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

        avg_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
        adv_loss = self.loss.compute(results['adv_y_pred'], results['g'].unsqueeze(1), return_dict=False)

        return avg_loss,  adv_loss

    def _update(self, results):
        """
        Computes the objective and updates the model.
        Also updates the results dictionary yielded by process_batch().
        Should be overridden to change algorithm update beyond modifying the objective.
        """
        # compute objective
        objective = self.objective(results)
        results['objective'] = objective[0].item()
        results['adv_objective'] = objective[1].item()
        # update
        self.model.zero_grad()
        self.adv_model.zero_grad()
        total_objective = objective[0] + objective[1]
        if self.scaler:
            self.scaler.scale(total_objective).backward()
        else:
            total_objective.backward()
        # update
        if self.max_grad_norm:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if self.max_grad_norm:
            clip_grad_norm_(self.adv_model.parameters(), self.max_grad_norm)
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.step(self.adv_optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            self.adv_optimizer.step()
        self.step_schedulers(
            is_epoch=False,
            metrics=results,
            log_access=False)
