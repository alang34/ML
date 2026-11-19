import torch


def getErrorDeficit(val_deficit, point_error):
    return torch.stack([val_deficit, point_error], dim=1)


def validation(model, val_inputs, val_targets, val_deficit, epoch, wake_threshold=0.05):
    pred = model(val_inputs)
    l2_error = torch.norm(pred - val_targets, dim=0) / torch.norm(val_targets, dim=0)
    point_error = torch.norm(pred - val_targets, dim=1)

    wake_mask = val_deficit > wake_threshold
    freestream_mask = ~wake_mask

    def segregateError(mask):
        if mask.sum() == 0:
            return torch.tensor(float("nan"),device=point_error.device)
        return point_error[mask].mean()

    wake_error = segregateError(wake_mask)
    freestream_error = segregateError(freestream_mask)
    overall_error = point_error.mean()

    row = torch.hstack([torch.tensor([epoch], dtype=torch.float32, device = val_inputs.device),l2_error,torch.stack([wake_error, freestream_error, overall_error])])

    extras = {
        "pred": pred.detach().cpu(),
        "target": val_targets.detach().cpu(),
        "error_deficit_pairs": getErrorDeficit(val_deficit, point_error).detach().cpu()}

    return row.detach().cpu(), extras