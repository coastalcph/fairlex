import torch
import numpy as np
from models.bert.bert import LongformerClassifier
from gates import DiffMaskGateInput
from getter_setter import bert_getter, bert_setter, f1_score


class MetaModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = LongformerClassifier.from_pretrained(model_path, num_labels=11)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.alpha = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.ones(()))
                for _ in range(self.model.config.num_hidden_layers + 2)
            ]
        )

        gate = DiffMaskGateInput

        self.gate = gate(
            hidden_size=self.model.config.hidden_size,
            hidden_attention=self.model.config.hidden_size // 4,
            num_hidden_layers=self.model.config.num_hidden_layers + 2,
            max_position_embeddings=1,
            placeholder=True,
            init_vector=self.model.longformer.embeddings.word_embeddings.weight[
                self.model.config.mask_token_id
            ],
        )

    def forward_explainer(
        self, input_ids, mask, layer_pred=None, attribution=False,
    ):

        inputs_dict = {
            "input_ids": input_ids,
            "mask": mask,
        }

        self.net.eval()

        (logits_orig,), hidden_states = bert_getter(self.model, inputs_dict, self.forward)

        if layer_pred is None:
            if self.hparams.stop_train:
                stop_train = (
                    lambda i: self.running_acc[i] > 0.75
                    and self.running_l0[i] < 0.1
                    and self.running_steps[i] > 100
                )
                p = np.array(
                    [0.1 if stop_train(i) else 1 for i in range(len(hidden_states))]
                )
                layer_pred = torch.tensor(
                    np.random.choice(range(len(hidden_states)), (), p=p / p.sum()),
                    device=input_ids.device,
                )
            else:
                layer_pred = torch.randint(len(hidden_states), ()).item()

        if "hidden" in self.hparams.gate:
            layer_drop = layer_pred
        else:
            layer_drop = 0

        (
            new_hidden_state,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
        ) = self.gate(
            hidden_states=hidden_states,
            mask=mask,
            layer_pred=None if attribution else layer_pred,
        )

        if attribution:
            return expected_L0_full
        else:
            new_hidden_states = (
                [None] * layer_drop
                + [new_hidden_state]
                + [None] * (len(hidden_states) - layer_drop - 1)
            )

            (logits,), _ = bert_setter(
                self.net, inputs_dict, new_hidden_states, self.forward
            )

        return (
            logits,
            logits_orig,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
            layer_drop,
            layer_pred,
        )

    def training_step(self, batch, batch_idx=None, optimizer_idx=None):
        input_ids, mask, labels = batch

        (
            logits,
            logits_orig,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
            layer_drop,
            layer_pred,
        ) = self.forward_explainer(input_ids, mask)

        loss_c = (
                torch.distributions.kl_divergence(
                    torch.distributions.Categorical(logits=logits_orig),
                    torch.distributions.Categorical(logits=logits),
                )
                - self.hparams.eps
        )

        loss_g = (expected_L0 * mask).sum(-1) / mask.sum(-1)

        loss = self.alpha[layer_pred] * loss_c + loss_g

        f1 = f1_score(logits.argmax(-1), logits_orig.argmax(-1))

        l0 = (expected_L0.exp() * mask).sum(-1) / mask.sum(-1)

        outputs_dict = {
            "loss_c": loss_c.mean(-1),
            "loss_g": loss_g.mean(-1),
            "alpha": self.alpha[layer_pred].mean(-1),
            "f1": f1,
            "l0": l0.mean(-1),
            "layer_pred": layer_pred,
            "r_acc": self.running_acc[layer_pred],
            "r_l0": self.running_l0[layer_pred],
            "r_steps": self.running_steps[layer_pred],
        }

        outputs_dict = {
            "loss": loss.mean(-1),
            **outputs_dict,
            "log": outputs_dict,
            "progress_bar": outputs_dict,
        }

        outputs_dict = {
            "{}{}".format("" if self.training else "val_", k): v
            for k, v in outputs_dict.items()
        }

        if self.training:
            self.running_acc[layer_pred] = (
                    self.running_acc[layer_pred] * 0.9 + f1 * 0.1
            )
            self.running_l0[layer_pred] = (
                    self.running_l0[layer_pred] * 0.9 + l0.mean(-1) * 0.1
            )
            self.running_steps[layer_pred] += 1

        return outputs_dict

    def validation_epoch_end(self, outputs):

        outputs_dict = {
            k: [e[k] for e in outputs if k in e]
            for k in ("val_loss_c", "val_loss_g", "val_f1", "val_l0")
        }

        outputs_dict = {k: sum(v) / len(v) for k, v in outputs_dict.items()}

        outputs_dict["val_loss_c"] += self.hparams.eps

        outputs_dict = {
            "val_loss": outputs_dict["val_l0"]
            if outputs_dict["val_loss_c"] <= self.hparams.eps_valid
               and outputs_dict["val_f1"] >= self.hparams.acc_valid
            else torch.full_like(outputs_dict["val_l0"], float("inf")),
            **outputs_dict,
            "log": outputs_dict,
        }

        return outputs_dict

    def optimizer_step(
        self,
        current_epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
    ):
        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

        elif optimizer_idx == 1:
            for i in range(len(self.alpha)):
                if self.alpha[i].grad:
                    self.alpha[i].grad *= -1

            optimizer.step()
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

            for i in range(len(self.alpha)):
                self.alpha[i].data = torch.where(
                    self.alpha[i].data < 0,
                    torch.full_like(self.alpha[i].data, 0),
                    self.alpha[i].data,
                )
                self.alpha[i].data = torch.where(
                    self.alpha[i].data > 200,
                    torch.full_like(self.alpha[i].data, 200),
                    self.alpha[i].data,
                )