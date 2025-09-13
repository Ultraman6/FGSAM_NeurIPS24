import torch


class SALP(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        base_optimizer,
        rho=0.05,
        rho_min=0.01,
        rho_max=1.0,
        rho_lr=1.0,
        adaptive=False,
        **kwargs
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(
            rho=rho,
            adaptive=adaptive,
            rho_min=rho_min,
            rho_max=rho_max,
            rho_lr=rho_lr,
            **kwargs
        )
        super(SALP, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self.rho_lr = rho_lr
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.g_0_norm = None
        self.g_1_loss = None

        for group in self.param_groups:
            init_rho = group.get("rho", rho)
            for p in group["params"]:
                state = self.state[p]
                if "rho" not in state:
                    state["rho"] = torch.full_like(p, init_rho)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        self.g_0_norm = self._grad_norm()

        for group in self.param_groups:
            adaptive = group.get("adaptive", False)
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["old_p"] = p.data.clone()
                state["g_0"] = p.grad.data.clone()
                rho_t = state["rho"]

                scale = 1.0 / (self.g_0_norm + 1e-12)
                weight = torch.pow(p, 2) if adaptive else 1.0
                e_w = weight * p.grad * (rho_t * scale)

                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False, g1_loss=None):
        if g1_loss is not None:
            if not torch.is_tensor(g1_loss):
                device = self.param_groups[0]["params"][0].device
                dtype = self.param_groups[0]["params"][0].dtype
                self.g_1_loss = torch.tensor(g1_loss, device=device, dtype=dtype)
            else:
                self.g_1_loss = g1_loss.detach()
        if self.g_1_loss is None:
            self.g_1_loss = torch.tensor(0.0, device=self.param_groups[0]["params"][0].device)

        g0_norm = (self.g_0_norm if self.g_0_norm is not None else torch.tensor(1.0, device=self.param_groups[0]["params"][0].device))
        g0_norm = g0_norm + 1e-12

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                g_1 = p.grad.data
                g_0 = state.get("g_0", None)
                if g_0 is not None:
                    h_0 = g_1 - g_0
                    rho_t = state["rho"]

                    rho_g = (g_1 * g_0) / (g0_norm ** 2) - (h_0 * g_0) * (self.g_1_loss / (g0_norm ** 3))

                    rho_t.add_(rho_g, alpha=self.rho_lr)
                    rho_t.clamp_(self.rho_min, self.rho_max)

                p.data = state["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SALP requires closure for step(), or use first_step/second_step explicitly."

        loss_origin = closure()
        loss_origin.backward()
        self.first_step(zero_grad=True)

        loss_perturbed = closure()
        loss_perturbed.backward()
        self.second_step(zero_grad=True, g1_loss=loss_perturbed.detach())

        return loss_origin, loss_perturbed

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            adaptive = group.get("adaptive", False)
            for p in group["params"]:
                if p.grad is None:
                    continue
                weight = torch.abs(p) if adaptive else 1.0
                norms.append((weight * p.grad).norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def get_rho(self, flatten=False):
        rhos = []
        for group in self.param_groups:
            for p in group["params"]:
                if "rho" in self.state[p]:
                    rp = self.state[p]["rho"]
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

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


