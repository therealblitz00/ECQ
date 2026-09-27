"""Wide & Deep multi-label network (André's Sprint 2 branch).

logit_k(x) = wide_k(x) + deep_k(x)
  wide : Linear(n_crm -> n_bil)       -- one additive weight per (pack, bill item):
                                          the "billing = union of per-pack items" structure
                                          found in Sprint 1 §8b / Sprint 2 §4.
  deep : n_crm -> hidden -> n_bil     -- a SHARED hidden layer: a rare pack's
                                          representation is learned from the gradients
                                          of all 731 outputs at once (multi-task pooling),
                                          and it can express pack interactions.
One model, 731 sigmoid outputs, BCE loss; early-stopped on inner-holdout EMR
(the competition metric), not on loss.

Why this rather than a second GBM: per-label LightGBM with min_child_samples=20
cannot isolate a pack that occurs in < 20 unique configurations, which is where
the catastrophic validation rows are. A linear path has no such floor, and the
shared layer pools evidence across labels.

Refs: Cheng et al. 2016, Wide & Deep Learning (arXiv:1606.07792);
      Nam et al. 2014, Large-scale multi-label text classification -- revisiting
      neural networks (arXiv:1312.5419): BCE + shared hidden layer is a strong
      multi-label baseline; Caruana 1997, Multitask Learning.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn


class _Net(nn.Module):
    def __init__(self, n_in, n_out, hidden, dropout, wide, prior_logit):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.deep = nn.Sequential(*layers, nn.Linear(d, n_out))
        self.wide = nn.Linear(n_in, n_out, bias=False) if wide else None
        with torch.no_grad():                      # start at the label prevalences
            self.deep[-1].bias.copy_(prior_logit)
            if self.wide is not None:
                self.wide.weight.zero_()

    def forward(self, x):
        z = self.deep(x)
        return z + self.wide(x) if self.wide is not None else z


class WideDeepMultiLabel:
    def __init__(self, hidden=(512,), dropout=0.2, wide=True, lr=2e-3, weight_decay=1e-4,
                 wide_lr_mult=10.0, lr_patience=None, batch_size=256, max_epochs=80, patience=8, val_frac=0.1,
                 seed=0, n_threads=None, verbose=True):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    # sklearn-style so Optuna / cv.run_cv can rebuild it from params
    def get_params(self):
        return {k: v for k, v in self.__dict__.items() if not k.endswith("_")}

    def fit(self, X, Y):
        if self.n_threads:
            torch.set_num_threads(self.n_threads)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, np.uint8); Y = np.asarray(Y, np.uint8)
        # rows passed in by run_cv are unique configurations, so a random split
        # of them is already a grouped split
        idx = rng.permutation(len(X))
        n_val = max(1, int(self.val_frac * len(X)))
        va, tr = idx[:n_val], idx[n_val:]
        p = Y[tr].mean(0).clip(1e-5, 1 - 1e-5)
        prior = torch.tensor(np.log(p / (1 - p)), dtype=torch.float32)
        self.net_ = _Net(X.shape[1], Y.shape[1], self.hidden, self.dropout, self.wide, prior)
        # Adam moves each weight by ~lr per step and an additive OR needs logits of
        # about +-10, so the wide layer gets its own larger step and no weight decay
        # (with decay it stays under-fitted: 0.18 vs 0.68 EMR in the synthetic check).
        groups = [{"params": self.net_.deep.parameters(), "lr": self.lr,
                   "weight_decay": self.weight_decay}]
        if self.net_.wide is not None:
            groups.append({"params": self.net_.wide.parameters(),
                           "lr": self.lr * self.wide_lr_mult, "weight_decay": 0.0})
        opt = torch.optim.AdamW(groups)
        sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=self.lr_patience)
                 if self.lr_patience else None)
        lossf = nn.BCEWithLogitsLoss()
        Xt, Yt = torch.from_numpy(X[tr]), torch.from_numpy(Y[tr])
        best, best_state, bad, t0 = (-1.0, np.inf), None, 0, time.perf_counter()
        for ep in range(self.max_epochs):
            self.net_.train()
            perm = torch.from_numpy(rng.permutation(len(tr)))
            for s in range(0, len(tr), self.batch_size):
                b = perm[s:s + self.batch_size]
                opt.zero_grad(set_to_none=True)
                loss = lossf(self.net_(Xt[b].float()), Yt[b].float())
                loss.backward()
                opt.step()
            P = self._proba(X[va])
            emr = float(((P >= 0.5) == Y[va].astype(bool)).all(1).mean())
            bce = float(-(Y[va] * np.log(P + 1e-7) + (1 - Y[va]) * np.log(1 - P + 1e-7)).mean())
            if sched is not None:
                sched.step(bce)
            if (emr, -bce) > (best[0], -best[1]):
                best, best_state, bad, self.best_epoch_ = (emr, bce), copy.deepcopy(
                    self.net_.state_dict()), 0, ep
            else:
                bad += 1
            if self.verbose and (ep % 5 == 0 or bad == 0):
                print(f"  ep {ep:3d} inner-EMR {emr:.4f} bce {bce:.5f} "
                      f"({time.perf_counter() - t0:.0f}s){' *' if bad == 0 else ''}")
            if bad >= self.patience:
                break
        self.net_.load_state_dict(best_state)
        self.inner_emr_ = best[0]
        return self

    @torch.no_grad()
    def _proba(self, X, bs=4096):
        self.net_.eval()
        out = [torch.sigmoid(self.net_(torch.from_numpy(np.asarray(X[s:s + bs], np.uint8)).float()))
               for s in range(0, len(X), bs)]
        return torch.cat(out).numpy()

    def predict_proba(self, X):
        return self._proba(X)


class SeedAverage:
    """Average predict_proba of the same model trained with several seeds.
    Reduces the variance of any stochastic model (NN, bagged trees)."""

    def __init__(self, make_model, seeds=(0, 1, 2)):
        self.make_model, self.seeds = make_model, seeds

    def fit(self, X, Y):
        self.models_ = [self.make_model(s).fit(X, Y) for s in self.seeds]
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)
