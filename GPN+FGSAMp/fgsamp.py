import torch

class FGSAMp(torch.optim.Optimizer):

    def __init__(self, params, base_optimizer, k, alpha, rho=0.05, lam=0.5, num_pert=-1, topology_norm=False, **kwargs):
        defaults = dict(alpha=alpha, rho=rho, **kwargs)
        super(FGSAMp, self).__init__(params, defaults)

        self.k = k
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.lam = lam
        self.num_pert = len(self.param_groups) if num_pert == -1 else num_pert
        self.topology_norm = topology_norm  # topology grad, g_G

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @staticmethod
    def normalized(g):
        return g / g.norm(p=2)

    def step(self, t, forward, g_mlp, zero_grad=False):
        # for t: 0 --> T-1

        if t % self.k == 0:  # including the first time
            grad_norm = self._grad_norm()
            for i, group in enumerate(self.param_groups[:self.num_pert]):
                scale = group['rho'] / (grad_norm + 1e-8)  # ρ/||▽wLs(w)||

                for j, p in enumerate(group['params']):
                    if p.grad is None:
                        continue

                    self.state[p]['old_p'] = p.data.clone()
                    self.state[p]['old_p_grag_gnn'] = p.grad.clone()
                    self.state[p]['old_p_grad_mlp'] = g_mlp[i][j]  # g

                    with torch.no_grad():
                        e_w = p.grad * scale.to(p)
                        # p.add_(e_w)
                        p.data = p.data + e_w
                        p.grad.data = (1. - self.lam) * p.grad.data

            # if zero_grad: self.zero_grad()
            loss_perturbed_mlp = forward(False)[0]
            (self.lam * loss_perturbed_mlp).backward()

        for group in self.param_groups[:self.num_pert]:
            for p in group['params']:
                if p.grad is None:
                    continue
                if t % self.k == 0:  # including the first time
                    old_p_grad_mlp = self.state[p]['old_p_grad_mlp']  # gmlp
                    g_mlp_grad_norm = FGSAMp.normalized(old_p_grad_mlp)  # gmlp/||gmlp||
                    gs_grad_norm = FGSAMp.normalized(p.grad)    # gs/||gs||
                    self.state[p]['gv'] = torch.sub(p.grad, p.grad.norm(p=2) * torch.sum(
                        g_mlp_grad_norm * gs_grad_norm) * g_mlp_grad_norm)  # gs - ||gs|| * (gmlp/||gmlp|| * gs/||gs||) * gmlp/||gmlp||

                    if self.topology_norm:
                        g_gnn = self.state[p]['old_p_grag_gnn']
                        g_gnn_grad_norm = FGSAMp.normalized(g_gnn)
                        self.state[p]['g_topo'] = torch.sub(g_gnn, g_gnn.norm(p=2) * torch.sum(
                            g_gnn_grad_norm * g_mlp_grad_norm) * g_mlp_grad_norm)  # ggnn - ||ggnn|| * (ggnn/||ggnn|| * gmlp/||gmlp||) * gmlp/||gmlp||

                else:
                    with torch.no_grad():
                        gv = self.state[p]['gv']
                        p.grad.add_(self.alpha.to(p) * (p.grad.norm(p=2) / (gv.norm(p=2) + 1e-8) * gv))

                        if self.topology_norm:
                            g_mlp = self.state[p]['old_p_grad_mlp']
                            g_gnn = g_mlp + (g_mlp.norm(p=2) / self.state[p]['g_topo'].norm(p=2) + 1e-8) * self.state[p]['g_topo']
                            p.grad.add_(self.lam * g_gnn)

                p.data = self.state[p]['old_p']

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

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
