import math

import pandas as pd
import torch
import torch.nn.functional as F

from data import SEED, set_seed
from metrics import evaluate
from models import Adversary, GradientReversal, Predictor


def projected_gradients(task_grads, adversary_grads, alpha):
    """Project over the concatenated parameter space, without building a flat tensor."""
    dot = sum((gp * ga).sum() for gp, ga in zip(task_grads, adversary_grads))
    norm_sq = sum(ga.square().sum() for ga in adversary_grads)
    coefficient = dot / norm_sq.clamp_min(torch.finfo(norm_sq.dtype).tiny)
    return [gp - (coefficient + alpha) * ga for gp, ga in zip(task_grads, adversary_grads)]


def train(data, mode="eq_odds", alpha=1.0, debias=True, epochs=20, batch_size=128,
          lr_p=1e-3, lr_a=1e-2, seed=SEED, verbose=True, adversary_hidden=0,
          schedule="constant", update="projection", representation=False):
    set_seed(seed)
    device = data["device"]
    predictor = Predictor(data["n_features"]).to(device)
    adversary = Adversary(mode, adversary_hidden, 64 if representation else None).to(device) if debias else None
    opt_p = torch.optim.Adam(predictor.parameters(), lr=lr_p)
    opt_a = torch.optim.Adam(adversary.parameters(), lr=lr_a) if adversary is not None else None
    sched_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=epochs)
    p_params = tuple(predictor.parameters())
    shuffle = torch.Generator().manual_seed(seed)
    x, y, z = data["train_ds"].tensors
    n = len(y)
    total_steps = epochs * math.ceil(n / batch_size)
    history = []
    step = 0

    for epoch in range(1, epochs + 1):
        predictor.train()
        loss_p_sum = loss_a_sum = 0.0
        adversary_count = 0
        for idx in torch.randperm(n, generator=shuffle).split(batch_size):
            step += 1
            weight = alpha * math.sqrt(step / total_steps) if schedule == "sqrt" else alpha
            xb, yb, zb = x[idx], y[idx].to(device), z[idx].to(device)
            if representation:
                logits, signal = predictor.with_hidden(xb)
            else:
                logits = signal = predictor(xb)
            loss_p = F.binary_cross_entropy_with_logits(logits, yb)
            opt_p.zero_grad(set_to_none=True)
            mask = yb == 1 if mode == "eq_opp" else slice(None)
            y_a, z_a = yb[mask], zb[mask]
            if adversary is None or len(y_a) == 0:
                loss_p.backward()
            else:
                opt_a.zero_grad(set_to_none=True)
                signal = signal[mask]
                if update == "grl":
                    loss_a = F.binary_cross_entropy_with_logits(
                        adversary(GradientReversal.apply(signal, weight), y_a), z_a)
                    (loss_p + loss_a).backward()
                else:
                    loss_a = F.binary_cross_entropy_with_logits(adversary(signal, y_a), z_a)
                    gp = torch.autograd.grad(loss_p, p_params, retain_graph=True)
                    ga = torch.autograd.grad(loss_a, p_params, retain_graph=True, allow_unused=True)
                    ga = [g if g is not None else torch.zeros_like(p) for p, g in zip(p_params, ga)]
                    for param, gradient in zip(p_params, projected_gradients(gp, ga, weight)):
                        param.grad = gradient
                    # Only the adversary receives the ordinary L_A gradient.
                    loss_a.backward(inputs=tuple(adversary.parameters()))
                opt_a.step()
                loss_a_sum += loss_a.item() * len(z_a)
                adversary_count += len(z_a)
            opt_p.step()
            loss_p_sum += loss_p.item() * len(idx)

        sched_p.step()
        metrics = evaluate(predictor, data["X_val"], data["y_val"], data["z_val"])
        row = {"epoch": epoch, "loss_P": loss_p_sum / n,
               "loss_A": loss_a_sum / adversary_count if adversary_count else float("nan"),
               "alpha": weight if debias else 0.0, **metrics}
        history.append(row)
        if verbose:
            print(f"epoch {epoch:2d} | L_P {row['loss_P']:.4f} | L_A {row['loss_A']:.4f} | "
                  f"val acc {metrics['accuracy']:.4f} | DP {metrics['DP gap']:.4f} | "
                  f"TPR {metrics['TPR gap']:.4f} | FPR {metrics['FPR gap']:.4f}", flush=True)
    return predictor, pd.DataFrame(history)
