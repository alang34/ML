import torch
import torch.nn.functional as F
import math
from config import params


def _grad(output, inputs):
    return torch.autograd.grad(
        outputs=output,
        inputs=inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]

def streamwise_vorticity(model, inputs, *, create_graph):
    """Return omega_x * L / Uref from the model's scaled curl velocity."""
    with torch.enable_grad():
        spatial_inputs = inputs.detach().requires_grad_(True)
        prediction = model(spatial_inputs)
        grad_v = torch.autograd.grad(
            prediction[:, 1].sum(), spatial_inputs, create_graph=create_graph, retain_graph=True,
        )[0]
        grad_w = torch.autograd.grad(
            prediction[:, 2].sum(), spatial_inputs, create_graph=create_graph,
        )[0]
    return model.crossflow_ratio * (grad_w[:, 1] - grad_v[:, 2])


def navier_stokes_residual(model, inputs):

    if not inputs.requires_grad:
        inputs = inputs.clone().detach().requires_grad_(True)

    pred = model(inputs)

    u_star = pred[:, 0:1]
    v_star = pred[:, 1:2]
    w_star = pred[:, 2:3]
    p_star = pred[:, 3:4]

    u_ref_scale = max(
        float(case["u_ref"])
        for case in params.CASE_FILES
    )

    u_ref = inputs[:, 3:4] * u_ref_scale

    u = u_star * u_ref
    v = v_star * u_ref
    w = w_star * u_ref

    p = p_star * (u_ref ** 2)

    l_ref = float(params.CASE_FILES[0]["l_ref"])
    cross_ref = float(params.CASE_FILES[0]["cross_ref"])

    mu = float(params.DYNAMIC_VISCOSITY)
    rho = float(params.CASE_FILES[0]["rho_ref"])
    nu = mu / rho

    grad_u = _grad(u, inputs)
    grad_v = _grad(v, inputs)
    grad_w = _grad(w, inputs)
    grad_p = _grad(p, inputs)


    u_x = grad_u[:, 0:1] / l_ref
    u_y = grad_u[:, 1:2] / cross_ref
    u_z = grad_u[:, 2:3] / cross_ref

    v_x = grad_v[:, 0:1] / l_ref
    v_y = grad_v[:, 1:2] / cross_ref
    v_z = grad_v[:, 2:3] / cross_ref

    w_x = grad_w[:, 0:1] / l_ref
    w_y = grad_w[:, 1:2] / cross_ref
    w_z = grad_w[:, 2:3] / cross_ref

    p_x = grad_p[:, 0:1] / l_ref
    p_y = grad_p[:, 1:2] / cross_ref
    p_z = grad_p[:, 2:3] / cross_ref


    grad_u_x = _grad(u_x, inputs)
    grad_u_y = _grad(u_y, inputs)
    grad_u_z = _grad(u_z, inputs)

    u_xx = grad_u_x[:, 0:1] / l_ref
    u_yy = grad_u_y[:, 1:2] / cross_ref
    u_zz = grad_u_z[:, 2:3] / cross_ref

    grad_v_x = _grad(v_x, inputs)
    grad_v_y = _grad(v_y, inputs)
    grad_v_z = _grad(v_z, inputs)

    v_xx = grad_v_x[:, 0:1] / l_ref
    v_yy = grad_v_y[:, 1:2] / cross_ref
    v_zz = grad_v_z[:, 2:3] / cross_ref

    grad_w_x = _grad(w_x, inputs)
    grad_w_y = _grad(w_y, inputs)
    grad_w_z = _grad(w_z, inputs)

    w_xx = grad_w_x[:, 0:1] / l_ref
    w_yy = grad_w_y[:, 1:2] / cross_ref
    w_zz = grad_w_z[:, 2:3] / cross_ref
    visc_x = nu * (u_xx +u_yy +u_zz)
    visc_y = nu * (v_xx +v_yy +v_zz)
    visc_z = nu * (w_xx +w_yy +w_zz)
    r_x = (u * u_x +v * u_y +w * u_z +p_x -visc_x)
    r_y = (u * v_x +v * v_y +w * v_z +p_y -visc_y)
    r_z = (u * w_x +v * w_y +w * w_z +p_z - visc_z)
    r_c = (u_x + v_y + w_z)
    r_x = r_x / ((u_ref ** 2) / l_ref)
    r_y = r_y / ((u_ref ** 2) / cross_ref)
    r_z = r_z / ((u_ref ** 2) / cross_ref)
    r_c = r_c / (u_ref / l_ref)
    residual = torch.cat([r_x, r_y, r_z, r_c],dim=1)

    return residual

def pde_component_rms(residual):
    rx_rms = torch.sqrt(torch.mean(residual[:, 0] ** 2))
    ry_rms = torch.sqrt(torch.mean(residual[:, 1] ** 2))
    rz_rms = torch.sqrt(torch.mean(residual[:, 2] ** 2))
    rc_rms = torch.sqrt(torch.mean(residual[:, 3] ** 2))

    return rx_rms, ry_rms, rz_rms, rc_rms

def pde_loss(residual):
    # Continuity is enforced by the curl outputs; retain it only for diagnostics.
    momentum = residual[:, :3]
    return F.smooth_l1_loss(momentum,torch.zeros_like(momentum),beta=5.0)

class LossController:
    def __init__(self, u_loss_fn, v_loss_fn, w_loss_fn, p_loss_fn, pde_loss_fn):
        self.lambda_u = 1.0
        self.lambda_v = 1.0
        self.lambda_w = 1.0
        self.lambda_p = 1.0
        self.lambda_pde = 0.0
        self.loss_history = []
        self.u_loss_fn = u_loss_fn
        self.v_loss_fn = v_loss_fn
        self.w_loss_fn = w_loss_fn
        self.p_loss_fn = p_loss_fn
        self.pde_loss_fn = pde_loss_fn
        self.target_scales = None
        self.window = 200
        self.status = -1
        self.pde_grad_ratio = 0.0
        self.gradient_cosine = 0.0
        self.data_grad_norm = 0.0
        self.pde_grad_norm = 0.0
        self.pde_ratio_cap = self.pde_cosine_gate = self.pde_desired_ratio = 0.0
        self.latest_data_loss = self.latest_pde_loss = self.latest_weighted_pde_loss = self.latest_total_loss = 0.0
        self.latest_boundary_loss = self.latest_weighted_boundary_loss = 0.0
        self.latest_vorticity_loss = self.latest_weighted_vorticity_loss = 0.0
        self.latest_objective_gradients = None
    def set_target_scales(self, target_scales):
        if target_scales.shape != (4,):
            raise ValueError(
                "target_scales must have shape (4,) in [u, v, w, p] order, "
                f"got {tuple(target_scales.shape)}"
            )
        self.target_scales = target_scales.detach()

    def recordLosses(self, pred, target, pde_residual):
        if self.target_scales is None:
            raise RuntimeError("Supervised target scales have not been configured")

        scaled_error = (pred - target) / self.target_scales
        zero_error = torch.zeros_like(scaled_error)
        u_loss = self.u_loss_fn(scaled_error[:, 0], zero_error[:, 0])
        v_loss = self.v_loss_fn(scaled_error[:, 1], zero_error[:, 1])
        w_loss = self.w_loss_fn(scaled_error[:, 2], zero_error[:, 2])
        p_loss = self.p_loss_fn(scaled_error[:, 3], zero_error[:, 3])
        pde_loss = self.pde_loss_fn(pde_residual)

        diagnostic_u_loss = self.u_loss_fn(pred[:, 0], target[:, 0])
        diagnostic_v_loss = self.v_loss_fn(pred[:, 1], target[:, 1])
        diagnostic_w_loss = self.w_loss_fn(pred[:, 2], target[:, 2])
        diagnostic_p_loss = self.p_loss_fn(pred[:, 3], target[:, 3])

        self.loss_history.append((
            diagnostic_u_loss.detach().item(), diagnostic_v_loss.detach().item(),
            diagnostic_w_loss.detach().item(), diagnostic_p_loss.detach().item(),
            pde_loss.detach().item(),
            self.lambda_u, self.lambda_v, self.lambda_w, self.lambda_p, self.lambda_pde,
            (
                u_loss + v_loss + w_loss + p_loss + self.lambda_pde*pde_loss
            ).detach().item(),
        ))
        self.loss_history = self.loss_history[-self.window:]
        
        return u_loss, v_loss, w_loss, p_loss, pde_loss
    def update(self, pred, target, pde_residual, epoch, model, boundary_loss=None, boundary_weight=0.0,
               vorticity_loss=None, vorticity_weight=0.0, measure_gradients=False):
        u_loss, v_loss, w_loss, p_loss, pde_loss = self.recordLosses(pred, target, pde_residual)
        data_loss = u_loss + v_loss + w_loss + p_loss
        if epoch <=10:
            self.status = -1
        elif epoch<=params.PRETRAIN_EPOCHS:
            self.status = 0
        else:
            self.status = 1

        if self.status == -1 or self.status == 0:
            self.lambda_u = 1.0
            self.lambda_v = 1.0
            self.lambda_w = 1.0
            self.lambda_p = 1.0
            self.lambda_pde = 0.0
        else:
            self.lambda_u = self.lambda_v = self.lambda_w = self.lambda_p = 1.0
            if epoch % params.PDE_GRADIENT_EVERY == 0:
                self.updatePDElambda(data_loss, pde_loss, model, epoch)
        terms = {"data": data_loss}
        if self.lambda_pde:
            terms["momentum"] = self.lambda_pde * pde_loss
        self.latest_boundary_loss = boundary_loss.detach().item() if boundary_loss is not None else 0.0
        self.latest_weighted_boundary_loss = boundary_weight * self.latest_boundary_loss
        if boundary_loss is not None and boundary_weight:
            terms["boundary"] = boundary_weight * boundary_loss
        self.latest_vorticity_loss = vorticity_loss.detach().item() if vorticity_loss is not None else 0.0
        self.latest_weighted_vorticity_loss = vorticity_weight * self.latest_vorticity_loss
        if vorticity_loss is not None and vorticity_weight:
            terms["vorticity"] = vorticity_weight * vorticity_loss
        total_loss = sum(terms.values())
        if measure_gradients:
            self.measure_objective_gradients(terms, model, epoch)
        
        self.latest_data_loss, self.latest_pde_loss = data_loss.detach().item(), pde_loss.detach().item()
        
        self.latest_weighted_pde_loss = self.lambda_pde*self.latest_pde_loss
        
        self.latest_total_loss = total_loss.detach().item()
        self.loss_history[-1] = self.loss_history[-1][:5] + (
            self.lambda_u, self.lambda_v, self.lambda_w, self.lambda_p,
            self.lambda_pde, self.latest_total_loss,
        )
        
        return total_loss

    def measure_objective_gradients(self, terms, model, epoch):
        parameters = [p for p in model.parameters() if p.requires_grad]
        norms = {}
        for name, term in terms.items():
            gradients = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
            squared_norm = sum((g.detach().square().sum() for g in gradients if g is not None),
                               term.new_zeros(()))
            norms[name] = squared_norm.sqrt()
        values = dict(zip(norms, torch.stack(list(norms.values())).cpu().tolist()))
        self.latest_objective_gradients = {"epoch": epoch, "norms": values}

    def updatePDElambda(self, data_loss, pde_loss, model, epoch):
        parameters = [p for p in model.parameters() if p.requires_grad]
        data_grads = torch.autograd.grad(data_loss, parameters, retain_graph=True, allow_unused=True)
        pde_grads = torch.autograd.grad(pde_loss, parameters, retain_graph=True, allow_unused=True)
        pairs = [(gd, gp) for gd, gp in zip(data_grads, pde_grads) if gd is not None and gp is not None]
        if not pairs:
            return
        data_norm = torch.sqrt(sum(gd.square().sum() for gd, _ in pairs))
        pde_norm = torch.sqrt(sum(gp.square().sum() for _, gp in pairs))
        cosine = sum((gd*gp).sum() for gd, gp in pairs)/(data_norm*pde_norm).clamp_min(1e-12)
        self.gradient_cosine = 0.95*self.gradient_cosine + 0.05*cosine.detach().item()
        self.data_grad_norm, self.pde_grad_norm = data_norm.detach().item(), pde_norm.detach().item()

        progress = min(1.0, (epoch-params.PRETRAIN_EPOCHS)/params.PDE_RAMP_EPOCHS)
        self.pde_ratio_cap = params.PDE_MAX_GRAD_RATIO*0.5*(1.0-math.cos(math.pi*progress))
        self.pde_cosine_gate = max(0.0, min(1.0, (self.gradient_cosine+0.05)/0.15))
        self.pde_desired_ratio = self.pde_ratio_cap*self.pde_cosine_gate
        self.pde_grad_ratio = 0.95*self.pde_grad_ratio + 0.05*self.pde_desired_ratio
        lambda_target = self.pde_grad_ratio*self.data_grad_norm/max(self.pde_grad_norm, 1e-12)
        lambda_smoothed = 0.95*self.lambda_pde + 0.05*lambda_target
        if self.lambda_pde > 0:
            lambda_smoothed = min(lambda_smoothed, params.PDE_LAMBDA_MAX_GROWTH_FACTOR*self.lambda_pde)
        self.lambda_pde = max(0.0, min(lambda_smoothed, lambda_target))
    
    def pde_stability(self, loss_history, window):
        recent = torch.tensor([l[4] for l in loss_history[-window:]])
        mean = recent.mean()
        if len(recent) < 2:
            return 0.0, mean.item(), 0.0
        std = recent.std()
        cov = std / (mean + 1e-8)
        return cov.item(), mean.item(), std.item()
    def pde_trend(self, loss_history, window):
        recent = torch.tensor([l[4] for l in loss_history[-window:]])
        if len(recent) < 2:
            return 0.0
        steps = torch.arange(len(recent), dtype=torch.float32)
        steps_centered = steps - steps.mean()
        slope = (steps_centered * (recent - recent.mean())).sum() / (steps_centered**2).sum()
        normalized_slope = slope / (recent.mean() + 1e-8)
        return normalized_slope.item()
    def report_LL(self):
        return self.loss_history[-1]
