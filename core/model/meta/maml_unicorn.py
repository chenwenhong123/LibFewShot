import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from core.utils import accuracy
from .meta_model import MetaModel
from ..backbone.utils import convert_maml_module

def update_params(loss, params, acc_gradients, step_size=0.5, first_order=True):
    name_list, tensor_list = zip(*params.items())
    grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order, allow_unused=True)
    updated_params = OrderedDict()
    for name, param, grad in zip(name_list, tensor_list, grads):
        if grad is not None:
            updated_params[name] = param - step_size * grad
            # accumulate gradients for final updates
            if name == 'classifier.weight':
                acc_gradients[0] = acc_gradients[0] + grad
            if name == 'classifier.bias':
                acc_gradients[1] = acc_gradients[1] + grad
        else:
            updated_params[name] = param  # No update if grad is None

    return updated_params, acc_gradients

def inner_train_step(model, support_data, support_target, inner_param):
    """ Inner training step procedure. 
        Should accumulate and record the gradient"""
    updated_params = OrderedDict(model.named_parameters())
    acc_gradients = [torch.zeros_like(updated_params['classifier.weight']), torch.zeros_like(updated_params['classifier.bias'])]
    
    for _ in range(inner_param["iter"]):
        ypred = model.forward_output(support_data, params=updated_params)
        loss = F.cross_entropy(ypred, support_target)
        updated_params, acc_gradients = update_params(loss, updated_params, acc_gradients, step_size=inner_param["lr"], first_order=True)
    return updated_params, acc_gradients

class MAMLUnicorn(MetaModel):
    def __init__(self, inner_param, feat_dim, **kwargs):
        super(MAMLUnicorn, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(feat_dim, self.way_num)
        self.fcone = nn.Linear(feat_dim, 1)
        self.inner_param = inner_param

        convert_maml_module(self)

    def forward_output(self, x, params=None):
        out1 = self.emb_func(x)
        out1 = out1.view(out1.size(0), -1)  # Flatten the output
        if params is None:
            out2 = self.classifier(out1)
        else:
            out2 = F.linear(out1, params['classifier.weight'], params['classifier.bias'])
        return out2

    def set_forward(self, batch):
        image, global_target = batch  # unused global_target
        image = image.to(self.device)
        support_image, query_image, support_target, query_target = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            episode_query_target = query_target[i].reshape(-1)  # Ensure query target is correctly reshaped

            # Set initial classifier
            self.classifier.weight.data = self.fcone.weight.data.repeat(self.way_num, 1)
            self.classifier.bias.data = self.fcone.bias.data.repeat(self.way_num)

            # Update with gradient descent
            updated_params, acc_gradients = self.set_forward_adaptation(episode_support_image, episode_support_target)

            # Re-update with the initial classifier and the accumulated gradients
            updated_params['classifier.weight'] = self.fcone.weight.repeat(self.way_num, 1) - self.inner_param["lr"] * acc_gradients[0]
            updated_params['classifier.bias'] = self.fcone.bias.repeat(self.way_num) - self.inner_param["lr"] * acc_gradients[1]

            output = self.forward_output(episode_query_image, params=updated_params)
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc

    def set_forward_loss(self, batch):
        output, acc = self.set_forward(batch)
        query_target = batch[1].to(self.device)
        loss = self.loss_func(output, query_target.contiguous().view(-1))
        return output, acc, loss

    def set_forward_adaptation(self, support_set, support_target):
        return inner_train_step(self, support_set, support_target, self.inner_param) 