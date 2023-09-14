import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import Optimizer, Adam
from torch.optim.optimizer import required
from torch.distributions import MultivariateNormal
from typing import List, Optional
from torch import Tensor
import copy
import pickle
#from global_variable import *
from collections import defaultdict
from torch.profiler import profile, record_function, ProfilerActivity
import sys
from torch.linalg import vector_norm
#  sigma = 0.01, ell=0.01, rho=0.01,
class PGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, radius=1, perturb_interval = 15, tolerance = 1e-1, *, maximize: bool = False, foreach: Optional[bool] = None,
                 differentiable: bool = False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable, perturb_interval=perturb_interval, tolerance=tolerance, radius=radius)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

        self.t = 0
        for group in self.param_groups:
            self.state[id(group)] = defaultdict(dict)
            state = self.state[id(group)]
            state['t_perturb'] = 0
            # a = 1/tolerance**2 + ell/(rho*tolerance)**0.5
            # num_of_elements = 0
            # for p in group["params"]:
            #     num_of_elements += torch.numel(p) 
            # b = num_of_elements / tolerance**2
            # group['radius'] = tolerance * (1+min(a,b))**0.5

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        has_sparse_grad = False
        num_of_elements = 0
        flatten_grad = None
        dtype = None
        device = None
        for p in group['params']:
            dtype = p.dtype
            device = p.device
            if p.grad is not None:
                num_of_elements += torch.numel(p)    
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                if (flatten_grad is None):
                    flatten_grad = torch.flatten(p.grad)
                else:
                    flatten_grad = torch.cat(flatten_grad, torch.flatten(p.grad))
                
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])
        grad_norm = vector_norm(flatten_grad)

        state = self.state[id(group)]
        if (grad_norm <= group['tolerance'] and self.t - state['t_perturb'] > group['perturb_interval']):
            # noise = torch.randn(num_of_elements, dtype=dtype, device=device)
            # noise.mul_(group['radius'] / vector_norm(noise))
            noise = MultivariateNormal(loc = torch.zeros((num_of_elements,)), covariance_matrix = torch.eye(num_of_elements) * (group['radius']**2 / num_of_elements)).sample().to(device)
            state['t_perturb'] = self.t
        else:
            noise = torch.zeros(num_of_elements, dtype=dtype, device=device)
            
        return has_sparse_grad, noise


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            has_sparse_grad, noise = self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)
            

            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'],
                noise = noise)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
        self.t += 1
        return loss


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: Optional[bool] = None,
        noise = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # why must we be explicit about an if statement for torch.jit.is_scripting here?
        # because JIT can't handle Optionals nor fancy conditionals when scripting
        if not torch.jit.is_scripting():
            pass
        else:
            foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    func = _single_tensor_sgd

    func(params,
         d_p_list,
         momentum_buffer_list,
         weight_decay=weight_decay,
         momentum=momentum,
         lr=lr,
         dampening=dampening,
         nesterov=nesterov,
         has_sparse_grad=has_sparse_grad,
         maximize=maximize,
         noise = noise)

@torch.no_grad()
def _single_tensor_sgd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       noise = None,
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool):
    index = 0
    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
        param.add_(d_p + noise[index:index+torch.numel(param)].view(param.shape), alpha = -lr)
        index += torch.numel(param)


# def _multi_tensor_sgd(params: List[Tensor],
#                       grads: List[Tensor],
#                       momentum_buffer_list: List[Optional[Tensor]],
#                       *,
#                       weight_decay: float,
#                       momentum: float,
#                       lr: float,
#                       dampening: float,
#                       nesterov: bool,
#                       maximize: bool,
#                       has_sparse_grad: bool):

#     if len(params) == 0:
#         return

#     grouped_tensors = _group_tensors_by_device_and_dtype([params, grads, momentum_buffer_list], with_indices=True)
#     for device_params, device_grads, device_momentum_buffer_list, indices in grouped_tensors.values():
#         device_has_sparse_grad = any(grad.is_sparse for grad in device_grads)

#         if maximize:
#             device_grads = torch._foreach_neg(tuple(device_grads))  # type: ignore[assignment]

#         if weight_decay != 0:
#             device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)

#         if momentum != 0:
#             bufs = []

#             all_states_with_momentum_buffer = True
#             for i in range(len(device_momentum_buffer_list)):
#                 if device_momentum_buffer_list[i] is None:
#                     all_states_with_momentum_buffer = False
#                     break
#                 else:
#                     bufs.append(device_momentum_buffer_list[i])

#             if all_states_with_momentum_buffer:
#                 torch._foreach_mul_(bufs, momentum)
#                 torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
#             else:
#                 bufs = []
#                 for i in range(len(device_momentum_buffer_list)):
#                     if device_momentum_buffer_list[i] is None:
#                         buf = device_momentum_buffer_list[i] = momentum_buffer_list[indices[i]] = \
#                             torch.clone(device_grads[i]).detach()
#                     else:
#                         buf = device_momentum_buffer_list[i]
#                         buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)

#                     bufs.append(buf)

#             if nesterov:
#                 torch._foreach_add_(device_grads, bufs, alpha=momentum)
#             else:
#                 device_grads = bufs

#         if not device_has_sparse_grad:
#             torch._foreach_add_(device_params, device_grads, alpha=-lr)
#         else:
#             # foreach APIs don't support sparse
#             for i in range(len(device_params)):
#                 device_params[i].add_(device_grads[i], alpha=-lr)