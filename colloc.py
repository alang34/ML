import torch
from config import params

class Collocator:
    def __init__(self, lower, upper, model, pde_residual, pool=20000, refresh = params.COLLOCATION_REFRESH_EVERY, device = None):
        self.lower = lower
        self.upper = upper
        self.dim = lower.shape[0]
        self.model = model
        self.pde_residual = pde_residual
        self.pool = pool
        self.refresh = refresh
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lower = self.lower.to(self.device)
        self.upper = self.upper.to(self.device)
        self.poolCand = None
        self.probs = None
        self.lastRefresh = None
    def uniform(self, n):
        return torch.rand(n, self.dim, device = self.device) * (self.upper - self.lower) + self.lower
    def refreshPool(self):
        candidates = self.uniform(self.pool)
        candidates.requires_grad_(True)
        residual = self.pde_residual(self.model, candidates)
        
        weights = residual.detach().abs()
        weights = weights.sum(dim = tuple(range(1, weights.dim())))
        weights = weights.squeeze()
        weights = weights + 1e-8
        
        probs = weights/weights.sum()
        self.poolCand = candidates.detach()
        self.probs = probs
    def getRefresh(self, n, epoch = None):
        refreshCond = (self.poolCand is None or (epoch is not None and (self.lastRefresh is None or epoch - self.lastRefresh >= self.refresh)))
        if refreshCond:
            self.refreshPool()
            self.lastRefresh = epoch
        idx = torch.multinomial(self.probs, n, replacement = True)
        points = self.poolCand[idx].clone()
        return points