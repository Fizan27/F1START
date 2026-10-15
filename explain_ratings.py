# explain_ratings.py
#
# What it does: works out WHY each driver has the rating they have, for the
#   "Why this rating" panel on the website.
#   1. Which teammate battles moved the rating most, up and down. For each
#      pair of teammates (Senna and Prost, 1988 to 1989, say) it estimates how
#      much the driver's peak rating would change if that pair's results were
#      removed. Battles the driver was not in count too: if a rival's old
#      teammate is rated higher, that changes the rival, and the change
#      travels along the chain.
#   2. The chain of teammates linking the driver to today's grid: the
#      shortest route through shared teammates, preferring teammates who
#      raced together for longer.
#
#   How (1) is done without refitting 2,828 times. Near the best fit the
#   badness is close to a bowl shape (a quadratic), and for a bowl, removing
#   a set of results moves the bottom by an amount that can be written down
#   exactly: (curvature WITHOUT those results) inverse, times how hard those
#   results were pulling. The curvature without them is the full curvature
#   (the Hessian, whose inverse is the model's covariance) minus their own
#   small part, and the "Woodbury identity" turns that subtraction into a
#   tiny calculation per pairing, so no big matrix is inverted again.
#   It is still an approximation (the badness is only roughly a bowl), but
#   unlike the simpler one-step version it does not understate the effect of
#   a pairing that carries most of the information about a driver.
#
# What it reads: the same tables as model.py, and data/ratings.parquet.
# What it produces: web/public/data/explain.json.
# Which files use it: web/app/drivers/page.tsx and the assistant's
#   explain_rating tool.
#
# Run:  .venv\Scripts\python.exe explain_ratings.py     (about a minute)

import heapq
import json
from pathlib import Path

import polars as pl
import torch

import fit_ratings
import model

OUTPUT_FILE = Path("web/public/data/explain.json")
BATTLES_SHOWN = 4  # per direction
SMALLEST_SHOWN = 0.005  # battles that move the rating less than this are left out


def pull_and_curvature(fit: model.Fit) -> dict:
    """For every comparison: how hard it pulls on the skill gap, and how sharply.

    pull:      slope of that comparison's log likelihood with respect to the
               gap between the two drivers, at the best fit. Positive means
               "this result wants A rated higher relative to B".
    curvature: how quickly that slope changes (minus the second derivative):
               how much information the comparison carries about the gap.
    Each comparison depends only on its own gap, so both can be worked out
    for all comparisons at once.
    """
    data, scales = fit.data, fit.scales
    skill = model.effective_skill(model.split(data["unknowns"], data["layout"]), data["links"])
    noise_name = {"quali_gap": "quali_noise", "old_quali_gap": "old_quali_noise", "pace_gap": "pace_noise"}
    out = {}
    for kind, rows in data["results"].items():
        gap = (skill[rows["a"]] - skill[rows["b"]]).detach().requires_grad_(True)
        if kind in ("race_ahead", "quali_ahead"):
            slope = scales["race_slope"] if kind == "race_ahead" else model.AHEAD_SLOPE_PER_NOISE / scales["quali_noise"]
            chance = -torch.nn.functional.binary_cross_entropy_with_logits(slope * gap, rows["value"], reduction="none")
        else:
            factor = scales["pace_factor"] if kind == "pace_gap" else 1.0
            chance = model.heavy_tailed_log_chance(rows["value"], -factor * gap, scales[noise_name[kind]])
        total = (rows["weight"] * chance).sum()
        (pull,) = torch.autograd.grad(total, gap, create_graph=True)
        (bend,) = torch.autograd.grad(pull.sum(), gap)
        out[kind] = (pull.detach(), -bend.detach())
    return out


def pairing_columns(fit: model.Fit) -> tuple[list, list]:
    """Group every comparison by teammate pairing, and within that by season pair.

    Returns the pairings (DriverA, DriverB, first season, last season, races)
    and, for each, a list of columns (skill position of A, of B, total pull,
    total curvature): one column per season the two were teammates.
    """
    data = fit.data
    measured = pull_and_curvature(fit)
    parts = []
    for kind, rows in data["results"].items():
        pull, curvature = measured[kind]
        kind_rows = data["comparisons"].filter(pl.col(kind).is_not_null())
        parts.append(kind_rows.select("DriverA", "DriverB", "Year", "Round").with_columns(
            pl.Series("a", rows["a"].tolist()), pl.Series("b", rows["b"].tolist()),
            pl.Series("pull", pull.tolist()), pl.Series("curvature", curvature.tolist())))
    table = pl.concat(parts)
    pairings = table.group_by("DriverA", "DriverB").agg(
        pl.col("Year").min().alias("From"), pl.col("Year").max().alias("To"),
        pl.struct("Year", "Round").n_unique().alias("Races")).sort("DriverA", "DriverB")
    columns = table.group_by("DriverA", "DriverB", "a", "b").agg(pl.col("pull").sum(), pl.col("curvature").sum())
    by_pairing = {}
    for a_driver, b_driver, a, b, pull, curvature in columns.iter_rows():
        by_pairing.setdefault((a_driver, b_driver), []).append((a, b, pull, curvature))
    return pairings.rows(), [by_pairing[(row[0], row[1])] for row in pairings.rows()]


def battle_effects(fit: model.Fit, weights: torch.Tensor) -> tuple[torch.Tensor, list]:
    """How much each teammate pairing moves each driver's peak rating: pairings x drivers.

    The number is the rating WITH the pairing minus the rating WITHOUT it, so
    positive means the battle pushed the rating up. For one pairing with
    columns U (one per season: +1 for A's skill that season, -1 for B's),
    total pulls s and curvatures c, removing it moves every driver's peak by
        m . (s + (diag(1/c) - M)^-1 M s),   M = U' Cov U,   m = U' Cov w
    which is the Woodbury identity applied to (Hessian - U diag(c) U')^-1.
    """
    covariance = fit.covariance
    reach = covariance @ weights.T  # column j: how each unknown is tied to driver j's peak
    pairings, columns = pairing_columns(fit)
    layout, links = fit.layout, fit.data["links"]

    effects = torch.zeros(len(pairings), weights.shape[0], dtype=model.DTYPE, device=model.DEVICE)
    for row, cols in enumerate(columns):
        # U: one column per season, +1 where A's effective skill sits, -1 for B's.
        u = torch.zeros(layout["total"], len(cols), dtype=model.DTYPE, device=model.DEVICE)
        for j, (a, b, _, _) in enumerate(cols):
            for unknown in model.effective_positions(layout, links, a):
                u[unknown, j] += 1.0
            for unknown in model.effective_positions(layout, links, b):
                u[unknown, j] -= 1.0
        pull = torch.tensor([col[2] for col in cols], dtype=model.DTYPE, device=model.DEVICE)
        curvature = torch.tensor([col[3] for col in cols], dtype=model.DTYPE, device=model.DEVICE).clamp(min=1e-9)
        small = u.T @ covariance @ u  # M = U' Cov U, one row and column per season
        correction = torch.linalg.solve(torch.diag(1 / curvature) - small, small @ pull)
        moved = u.T @ reach  # m for every driver at once: seasons x drivers
        effects[row] = (pull + correction) @ moved
    return effects, pairings


def chains_to_today(pairs: pl.DataFrame, latest: int) -> dict:
    """For every driver, the chain of teammates to someone on today's grid.

    Shortest in links first; among equally short chains, the one whose
    teammates raced together most (each link costs 1 + 1 / races together).
    Worked out once, backwards from today's drivers, for everyone at once.
    """
    together = pairs.group_by("DriverA", "DriverB").agg(
        pl.len().alias("races"), pl.col("Year").min().alias("from"), pl.col("Year").max().alias("to"))
    links = {}
    for a, b, races, first, last in together.iter_rows():
        links.setdefault(a, []).append((b, races, first, last))
        links.setdefault(b, []).append((a, races, first, last))
    latest_pairs = pairs.filter(pl.col("Year") == latest)
    today = set(latest_pairs["DriverA"]) | set(latest_pairs["DriverB"])
    cost = {driver: 0.0 for driver in today}
    step = {}  # driver -> (next driver towards today, races, first, last)
    queue = [(0.0, driver) for driver in today]
    heapq.heapify(queue)
    while queue:
        so_far, driver = heapq.heappop(queue)
        if so_far > cost[driver]:
            continue
        for mate, races, first, last in links.get(driver, []):
            new = so_far + 1 + 1 / races
            if new < cost.get(mate, float("inf")):
                cost[mate] = new
                step[mate] = (driver, races, first, last)
                heapq.heappush(queue, (new, mate))
    chains = {}
    for driver in links:
        chain, here = [], driver
        while here in step:
            nxt, races, first, last = step[here]
            chain.append([nxt, first, last, races])
            here = nxt
        chains[driver] = chain
    return chains


def main():
    pairs, pace = model.load_tables()
    ratings = pl.read_parquet("data/ratings.parquet")
    latest = pairs["Year"].max()
    fit = model.fit_model(pairs, pace, latest)
    _, _, weights = fit_ratings.peak_ratings(fit)
    drivers = list(dict.fromkeys(driver for driver, _ in fit.index))  # same order as the weights
    effects, pairings = battle_effects(fit, weights)
    chains = chains_to_today(pairs, latest)

    explained = {}
    for column, driver in enumerate(drivers):
        order = effects[:, column].argsort()

        def battles(positions):
            shown = []
            for position in positions.tolist():
                value = effects[position, column].item()
                if abs(value) >= SMALLEST_SHOWN:
                    a, b, first, last, races = pairings[position]
                    shown.append([a, b, first, last, races, round(value, 3)])
            return shown

        explained[driver] = {
            "up": [b for b in battles(order.flip(0)[:BATTLES_SHOWN]) if b[5] > 0],
            "down": [b for b in battles(order[:BATTLES_SHOWN]) if b[5] < 0],
            "chain": chains.get(driver, []),
        }
    OUTPUT_FILE.write_text(json.dumps(explained, separators=(",", ":")), encoding="utf-8")
    print(f"Saved explanations for {len(explained)} drivers to {OUTPUT_FILE} "
          f"({OUTPUT_FILE.stat().st_size / 1024:.0f} KB)")
    names = dict(ratings.select("DriverId", "DriverName").iter_rows())
    for driver in ["hamilton", "michael_schumacher", "senna", "frentzen"]:
        print(f"\n{names.get(driver, driver)}: up {explained[driver]['up'][:3]}\n  down {explained[driver]['down'][:3]}"
              f"\n  chain {[link[0] for link in explained[driver]['chain']]}")


if __name__ == "__main__":
    main()
