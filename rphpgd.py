import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.optimizer import required
from typing import List, Optional
from collections import defaultdict, deque
from torch.linalg import vector_norm, norm
from torch.nn.utils import parameters_to_vector
from scipy.stats import ortho_group
from numpy.random import choice
from torch.distributions import MultivariateNormal

class RPHPGD(Optimizer):
    def __init_state__(self):
        for group in self.param_groups:
                # Each optimizer can have multiple parameter groups
                # Thus using id of group as key of state
                self.state[id(group)] = dict()
                state = self.state[id(group)]
                state['perturbation_radius'] = group['init_perturbation_radius']
                state['saddle'] = False
                state['grad_norm'] = 0
                state['ell'] = 0.01
                state['rho'] = 0.01
                # minimum eigenvalue in current iteration
                state['e_min'] = 0
                # maximum eigenvalue in current iteration
                state['e_max'] = 0
                state['prev_max_eigenvalue'] = 0
                state['t_inter'] = 0
                state['t_rp'] = 0
                state['grad_buffer'] = deque([], maxlen = group['buffer_size'])
                state['x_buffer'] = deque([], maxlen = group['buffer_size'])
                state['hessian_buffer'] = deque([], maxlen = group['buffer_size'])
                state['first'] = True
                
    def __init__(self, params, 
                 lr=required, 
                 momentum=0, 
                 dampening=0,
                 weight_decay=0, 
                 nesterov=False,
                 tolerance = 1e-2,
                 PGD_interval = 15,
                 RPHPGD_interval = 30,
                 subspace_dimension = 2,
                 init_perturbation_radius = 1,
                 sigma = 1e-2,
                 delta = 1e-5,
                 buffer_size = 3,
                 *, 
                 maximize: bool = False, 
                 foreach: Optional[bool] = None,
                 differentiable: bool = False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        # Init default parameters
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable, tolerance=tolerance, PGD_interval=PGD_interval,
                        RPHPGD_interval=RPHPGD_interval, subspace_dimension=subspace_dimension, init_perturbation_radius=init_perturbation_radius,
                        sigma=sigma, delta=delta, buffer_size=buffer_size)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)
        self.__init_state__()
        self.t = 0


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        has_sparse_grad = False
        num_of_element = 0
        grad_norm = 0
        gradient = None
        device = None
        dtype = None

        for p in group['params']:
            device = p.device
            dtype = p.dtype
            if p.grad is not None:
                num_of_element += torch.numel(p)
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True
                
                if (gradient is None):
                    gradient = torch.flatten(p.grad)
                else:
                    gradient = torch.cat((gradient, torch.flatten(p.grad)), dim=0)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

        state = self.state[id(group)]
        state['grad_0'] = gradient
        # Compute gradient norm using vector 2-norm
        grad_norm = vector_norm(gradient)
        state['grad_norm'] = grad_norm
        state['num_of_element'] = num_of_element

        return has_sparse_grad, grad_norm, num_of_element, device, dtype


    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        PGD_flag = False
        RPHPGD_flag = False
        finish = False
        e_max = None
        e_min = None
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the gradient(x+delta*d)
        """
        if closure is None:
            raise NotImplementedError("closure is None")

        for group in self.param_groups:
            state = self.state[id(group)]
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            tolerance = group['tolerance']

            has_sparse_grad, grad_norm, num_of_elements, device, dtype = self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

            # Init noise tensor to zero
            noise = torch.zeros(num_of_elements, dtype=dtype, device = device)
            
            if (grad_norm <= tolerance and (self.t - state['t_inter'] > group['PGD_interval'])):
                # This flag will be return and it indicates if PGD is triggered at this iteration
                # It can be removed if there is no need for plotting the triggered iterations
                PGD_flag = True

                # temp = torch.randn(num_of_elements, dtype=dtype, device=device)
                # temp.mul_(state['perturbation_radius'] / vector_norm(temp))

                # Sampling from multivariate normal distribution with covariance matrix = (r^2/n)I
                temp = MultivariateNormal(loc = torch.zeros((num_of_elements,)), covariance_matrix = torch.eye(num_of_elements) * (state['perturbation_radius']**2 / num_of_elements)).sample().to(device)
                noise.add_(temp)
                state['t_inter'] = self.t
            
            if (grad_norm <= tolerance and (self.t - state['t_rp'] > group['RPHPGD_interval'])):
                # If we tirggered RPHPGD for the first time, we need to compute the projected Hessian first for the eigenvalues and eigenvectors.
                if (state['first']):
                    state['first'] = False
                    compute_projected_hessian(state = state, group = group, closure = closure, kwargs = kwargs, device = device, dtype = dtype)

                e_max, e_min, Hessian_v_min = compute_projected_hessian(state = state, group = group, closure = closure, kwargs = kwargs, device = device, dtype = dtype)

                compute_l(state)

                compute_rho(state)

                update_perturbation_radius(state, group)
                
                if (e_min > 0):
                    state['t_rp'] = self.t
                else:
                    RPHPGD_flag = True
                    noise.add_(Hessian_v_min, alpha = -1/(e_max * group['lr'])) # Negative because lr is assigned to be negative in sgd function
                    state['first'] = True
                    state['t_rp'] = self.t
                    state['prev_max_eigenvalue'] = 0
                    
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
        return PGD_flag, RPHPGD_flag, grad_norm#, finish, e_max, e_min

def compute_projected_hessian(state, group, closure, kwargs, device, dtype):
    grad_0 = state['grad_0']
    delta = group['delta']
    subspace_dimension = group['subspace_dimension']
    omega_size = (subspace_dimension, torch.numel(grad_0))
    omega = torch.randn(omega_size, dtype = dtype, device = device)
    projected_Hessian = None

    # Calculate projected Hessian column by column
    for i in range(subspace_dimension):
        omega[i] = omega[i] / vector_norm(omega[i])
        with torch.enable_grad():
        # Compute hessian-vector product using closure
            grad_1 = closure(delta, omega[i], **kwargs)

        # Subtract the previous maximum eigenvalue
        temp = ((grad_1 - grad_0) / delta) - (state['prev_max_eigenvalue'] * torch.matmul(torch.eye(torch.numel(grad_0), dtype=dtype, device=device), omega[i].t()))
        temp = torch.unsqueeze(temp, dim=0)

        if (projected_Hessian is None):
            projected_Hessian = temp
        else:
            projected_Hessian = torch.cat((projected_Hessian, temp), dim = 0)

    projected_Hessian = projected_Hessian.t()
    omega = omega.t()

    ###### H = R * (Q^H * omega)^-1 ######
    Q, R = torch.linalg.qr(projected_Hessian)
    Q_T_omega = torch.matmul(Q.t(), omega)
    Q_T_omega_inv = torch.linalg.inv(Q_T_omega)
    Hessian_like = torch.matmul(R, Q_T_omega_inv)

    eigenvalues, eigenvectors = torch.linalg.eig(Hessian_like)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    e_min = eigenvalues.min()
    e_max = eigenvalues.max()

    min_index = torch.argmin(eigenvalues)
    v_min = eigenvectors[min_index]
    v_min = v_min.view(torch.numel(v_min),1)

    # Adding back the previous max eigenvalue
    e_min += state['prev_max_eigenvalue']
    e_max += state['prev_max_eigenvalue']

    # Not sure if this is correct or just assign state["prev_max_eigenvalue"] = e_max
    state['prev_max_eigenvalue'] = e_max if e_max > 0 else 0

    Hessian_v_min = torch.matmul(Q,v_min)
    Hessian_v_min = torch.flatten(Hessian_v_min)

    state['Hessian_v_min'] = Hessian_v_min # Ritz vector
    state['e_min'] = e_min # min eigenvalue

    state['e_max'] = e_max # max eigenvalue
    state['v_min'] = v_min # min eigenvector
    
    state['grad_buffer'].append(grad_0)
    state['x_buffer'].append(parameters_to_vector(group['params']))
    state['hessian_buffer'].append(Hessian_like)

    return e_max, e_min, Hessian_v_min

def compute_l(state):
    max_grad_difference = -1
    index_1 = -1
    index_2 = -1
    for i, p in enumerate(state['grad_buffer']):
        for j, pp in enumerate(state['grad_buffer']):
            if (i != j):
                temp = vector_norm(p - pp, ord = 2)
                if (temp > max_grad_difference):
                    max_grad_difference = temp
                    index_1 = i
                    index_2 = j
    grad_x_1 = state['x_buffer'][index_1]
    grad_x_2 = state['x_buffer'][index_2]
    state['ell'] = (max_grad_difference / vector_norm(grad_x_1 - grad_x_2)).item()

def compute_rho(state):
    max_hessian_difference = -1
    index_1 = -1
    index_2 = -1
    for i, p in enumerate(state['hessian_buffer']):
        for j, pp in enumerate(state['hessian_buffer']):
            if (i != j):
                temp = vector_norm(p - pp)
                if (temp > max_hessian_difference):
                    max_hessian_difference = temp
                    index_1 = i
                    index_2 = j
    hessian_x_1 = state['x_buffer'][index_1]
    hessian_x_2 = state['x_buffer'][index_2]
    state['rho'] = (max_hessian_difference / vector_norm(hessian_x_1 - hessian_x_2)).item()

def update_perturbation_radius(state, group):
    # Update r
    if state['rho'] > 0 and state['ell'] > 0:
        sigma = group['sigma']
        ell = state['ell']
        rho = state['rho']
        epsilon = group['tolerance']
        a = (sigma**2 / epsilon**2)+(ell / ((rho * epsilon)**0.5))
        b = (sigma**2 * torch.numel(state['grad_0'])) / epsilon**2
        r_parameter = 1 + min(a,b)
        state['perturbation_radius'] = epsilon * r_parameter**0.5

def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: Optional[bool] = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        noise: Tensor):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """
    # Comment part is of no use in our experiment

    # if foreach is None:
    #     # why must we be explicit about an if statement for torch.jit.is_scripting here?
    #     # because JIT can't handle Optionals nor fancy conditionals when scripting
    #     if not torch.jit.is_scripting():
    #         _, foreach = _default_to_fused_or_foreach(params, differentiable=False, use_fused=False)
    #     else:
    #         foreach = False

    # if foreach and torch.jit.is_scripting():
    #     raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    # if foreach and not torch.jit.is_scripting():
    #     func = _multi_tensor_sgd
    # else:
    #     func = _single_tensor_sgd
    
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

def _single_tensor_sgd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool,
                       noise: Tensor):

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

        # Adding corresponding part of noise to parameters
        param.add_(d_p + noise[index:index+torch.numel(param)].view(param.shape), alpha=-lr)
        index += torch.numel(param)


def _multi_tensor_sgd(params: List[Tensor],
                      grads: List[Tensor],
                      momentum_buffer_list: List[Optional[Tensor]],
                      *,
                      weight_decay: float,
                      momentum: float,
                      lr: float,
                      dampening: float,
                      nesterov: bool,
                      maximize: bool,
                      has_sparse_grad: bool):

    if len(params) == 0:
        return

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, momentum_buffer_list], with_indices=True)
    for ((device_params, device_grads, device_momentum_buffer_list), indices) in grouped_tensors.values():
        device_has_sparse_grad = any(grad.is_sparse for grad in device_grads)

        if maximize:
            device_grads = torch._foreach_neg(tuple(device_grads))  # type: ignore[assignment]

        if weight_decay != 0:
            device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)

        if momentum != 0:
            bufs = []

            all_states_with_momentum_buffer = True
            for i in range(len(device_momentum_buffer_list)):
                if device_momentum_buffer_list[i] is None:
                    all_states_with_momentum_buffer = False
                    break
                else:
                    bufs.append(device_momentum_buffer_list[i])

            if all_states_with_momentum_buffer:
                torch._foreach_mul_(bufs, momentum)
                torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
            else:
                bufs = []
                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        buf = device_momentum_buffer_list[i] = momentum_buffer_list[indices[i]] = \
                            torch.clone(device_grads[i]).detach()
                    else:
                        buf = device_momentum_buffer_list[i]
                        buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)

                    bufs.append(buf)

            if nesterov:
                torch._foreach_add_(device_grads, bufs, alpha=momentum)
            else:
                device_grads = bufs

        if not device_has_sparse_grad:
            torch._foreach_add_(device_params, device_grads, alpha=-lr)
        else:
            # foreach APIs don't support sparse
            for i in range(len(device_params)):
                device_params[i].add_(device_grads[i], alpha=-lr)