# sample_posterior.py
#
# What it does: checks the model's uncertainty with a method that makes far
#   fewer assumptions than the Laplace approximation in model.py.
#
#   The Laplace approximation assumes the uncertainty is a perfect bell
#   shape centred on the best fit, and treats the five scale numbers (how
#   noisy qualifying is, and so on) as known exactly. Here, a sampler called
#   NUTS (the "No-U-Turn Sampler", a form of Hamiltonian Monte Carlo) draws
#   thousands of plausible versions of EVERY unknown at once, including the
#   scales, straight from the model's own probability. If the real
#   uncertainty is lopsided, or wider because the scales are uncertain too,
#   the samples show it.
#
#   One trick makes it fast: the sampler works in coordinates where the
#   Laplace answer would be a plain round bell curve. If Laplace were exactly
#   right the samples would look like plain random noise in those
#   coordinates; wherever Laplace is wrong, the sampler is free to go there.
#   The Laplace answer is a map, not an assumption.
#
# What it reads: the same tables as model.py.
# What it produces: data/posterior_samples.pt (every sampled version of every
#   skill, wet adjustment and scale), and a summary printed to the screen.
# Which files use it: fit_ratings.py (peak ratings and their ranges come from
#   these samples when the file exists).
#
# Run:  .venv\Scripts\python.exe sample_posterior.py     (about 10 minutes on the GPU)

import time
from pathlib import Path

import polars as pl
import torch
from pyro.infer import MCMC, NUTS

import model

OUTPUT_FILE = Path("data/posterior_samples.pt")
SAMPLES = 1000
WARMUP = 500  # steps used only to tune the sampler's step size, then thrown away
SEED = 2026


def all_unknowns_hessian(data, unknowns, raw_scales, spread, drift) -> torch.Tensor:
    """The Hessian over everything, scales included (model.py leaves the scales out)."""
    count = unknowns.numel()

    def badness_of(values):
        return model.badness(values[:count], values[count:], data, spread, drift)

    return torch.autograd.functional.hessian(badness_of, torch.cat([unknowns, raw_scales]), vectorize=True)


def run_sampler(fit: model.Fit) -> torch.Tensor:
    """Draw samples of all unknowns and scales. Returns samples x (unknowns + scales)."""
    data = fit.data
    centre = torch.cat([data["unknowns"], data["raw_scales"]])
    hessian = all_unknowns_hessian(data, data["unknowns"], data["raw_scales"], fit.skill_spread, fit.season_drift)
    # The "map": turns plain random noise z into the Laplace bell curve.
    root = torch.linalg.cholesky(torch.linalg.inv(hessian))
    count = data["unknowns"].numel()

    def potential(params):
        values = centre + root @ params["z"]
        return model.badness(values[:count], values[count:], data, fit.skill_spread, fit.season_drift)

    torch.manual_seed(SEED)
    kernel = NUTS(potential_fn=potential, adapt_step_size=True, adapt_mass_matrix=False, max_tree_depth=8)
    sampler = MCMC(kernel, num_samples=SAMPLES, warmup_steps=WARMUP,
                   initial_params={"z": torch.zeros_like(centre)}, disable_progbar=False)
    sampler.run()
    z = sampler.get_samples()["z"]  # samples x unknowns
    return centre + z @ root.T


def effective_sample_size(chain: torch.Tensor) -> torch.Tensor:
    """Roughly how many independent draws a correlated chain is worth, per column.

    Successive samples are correlated; this counts how much, using the sum of
    the chain's autocorrelations until they turn negative.
    """
    chain = chain - chain.mean(dim=0)
    length = chain.shape[0]
    variance = (chain ** 2).mean(dim=0)
    total = torch.ones_like(variance)
    alive = torch.ones_like(variance, dtype=torch.bool)
    for lag in range(1, 200):
        rho = (chain[lag:] * chain[:-lag]).mean(dim=0) / variance
        alive &= rho > 0
        total += 2 * rho * alive
        if not alive.any():
            break
    return length / total


def main():
    pairs, pace = model.load_tables()
    last_year = pl.read_parquet("data/results.parquet")["Year"].max()
    started = time.time()
    fit = model.fit_model(pairs, pace, last_year, wet_races=model.load_wet_races())
    print(f"Fitted {fit.layout['total']} unknowns up to {last_year}; sampling {SAMPLES} versions with NUTS...")
    samples = run_sampler(fit)
    count = fit.layout["total"]
    laplace_sd = fit.covariance.diagonal().sqrt()
    sampled_sd = samples[:, :count].std(dim=0)
    start, end = fit.layout["skill"]
    ratio = (sampled_sd[start:end] / laplace_sd[start:end])
    ess = effective_sample_size(samples[:, start:end])
    print(f"Done in {(time.time() - started) / 60:.1f} minutes.")
    print(f"Skill uncertainty, sampled / Laplace: median {ratio.median():.2f}, "
          f"10th to 90th percentile {ratio.quantile(0.1):.2f} to {ratio.quantile(0.9):.2f}")
    print(f"Effective samples per skill: median {ess.median():.0f}, lowest {ess.min():.0f} of {SAMPLES}")
    scales = model.positive_scales(samples[:, count:].T)
    for name, values in scales.items():
        print(f"  {name:<16} {values.mean():.3f}  (90% range {values.quantile(0.05):.3f} to {values.quantile(0.95):.3f})")
    torch.save({
        "keys": list(fit.index), "drivers": fit.data["drivers"], "layout": fit.layout,
        "samples": samples[:, :count].float().cpu(), "scales": {k: v.float().cpu() for k, v in scales.items()},
        "laplace_sd": laplace_sd.float().cpu(), "effective_samples": ess.float().cpu(),
    }, OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
