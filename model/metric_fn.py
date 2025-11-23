import torch

def AEE(pred, gt):
    """
    pred, gt: [N,2]
    """
    return torch.norm(pred - gt, dim=1).mean()

def percent_outliers(pred, gt):
    epe = torch.norm(pred - gt, dim=1)
    gt_mag = torch.norm(gt, dim=1)

    threshold = torch.clamp(0.05 * gt_mag, min=3.0)
    return (epe > threshold).float().mean() * 100.0

def flow_accuracy(pred, gt, zeta=0.25):
    rel = torch.norm(pred - gt, dim=1) / (torch.norm(gt, dim=1) + 1e-12)
    return (rel < zeta).float().mean()
