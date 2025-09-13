import torch


class FGSALPp(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, k:int, alpha:float, rho:float=0.05, lam:float=0.5,
                 num_pert:int=-1, rho_min:float=0.01, rho_max:float=1.0, rho_lr:float=1.0, **kwargs):
        defaults = dict(alpha=alpha, rho=rho, **kwargs)
        super(FGSALPp, self).__init__(params, defaults)

        self.k = k
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.lam = lam
        self.num_pert = len(self.param_groups) if num_pert == -1 else num_pert

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

        self.rho_lr = rho_lr
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.g_0_norm = None
        self.g_1_loss = None

        for group in self.param_groups[:self.num_pert]:
            init_rho = group.get('rho', rho)
            for p in group['params']:
                state = self.state[p]
                if 'rho' not in state:
                    state['rho'] = torch.full_like(p, init_rho)

    @staticmethod
    def normalized(g):
        return g / (g.norm(p=2) + 1e-8)

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups[:self.num_pert]
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    @torch.no_grad()
    def first_step(self):
        self.g_0_norm = self._grad_norm()
        for i, group in enumerate(self.param_groups[:self.num_pert]):
            scale = group['rho'] / (self.g_0_norm + 1e-8)
            for j, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.state[p]
                state['old_p'] = p.data.clone()
                state['g_0'] = p.grad.data.clone()
                p.add_(p.grad * scale.to(p))

    @torch.no_grad()
    def adapt_rho_and_restore(self):
        g0_norm = (self.g_0_norm if self.g_0_norm is not None else torch.tensor(1.0, device=self.param_groups[0]['params'][0].device))
        g0_norm = g0_norm + 1e-8
        for group in self.param_groups[:self.num_pert]:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                g_1 = p.grad.data
                g_0 = state.get('g_0', None)
                if g_0 is not None:
                    h_0 = g_1 - g_0
                    rho_t = state['rho']
                    rho_g = (g_1 * g_0) / (g0_norm ** 2) - (h_0 * g_0) * (self.g_1_loss / (g0_norm ** 3))
                    rho_t.add_(rho_g, alpha=self.rho_lr)
                    rho_t.clamp_(self.rho_min, self.rho_max)
                p.data = state['old_p']

    def step(self, t:int, forward, g_mlp, zero_grad=False):
        if t % self.k == 0:
            grad_norm = self._grad_norm()
            for i, group in enumerate(self.param_groups[:self.num_pert]):
                scale = group['rho'] / (grad_norm + 1e-8)
                for j, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    self.state[p]['old_p'] = p.data.clone()
                    with torch.no_grad():
                        e_w = p.grad * scale.to(p)
                        p.data = p.data + e_w
                        p.grad.data = (1. - self.lam) * p.grad.data

            loss_perturbed_mlp = forward(False)[0]
            (self.lam * loss_perturbed_mlp).backward()

        for group in self.param_groups[:self.num_pert]:
            for p in group['params']:
                if p.grad is None:
                    continue
                if t % self.k == 0:
                    pass
                else:
                    with torch.no_grad():
                        p.grad.add_(0)
                p.data = self.state[p]['old_p']

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def get_rho(self, flatten=False):
        rhos = []
        for group in self.param_groups[:self.num_pert]:
            for p in group['params']:
                if 'rho' in self.state[p]:
                    rp = self.state[p]['rho']
                    rhos.append(rp.clone().flatten() if flatten else rp.clone())
        return rhos

    @torch.no_grad()
    def get_rho_stats(self):
        rhos = self.get_rho(flatten=True)
        if not rhos:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        all_rho = torch.cat(rhos, dim=0)
        return {
            "mean": all_rho.mean().item(),
            "std": all_rho.std(unbiased=False).item(),
            "min": all_rho.min().item(),
            "max": all_rho.max().item(),
        }


