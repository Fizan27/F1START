# app.py
#
# What it does: the website (Phase 4), built with Streamlit. Pick a season,
#   race and driver, then:
#     1. see the strategy the team really used, on a timeline
#     2. edit it (one or two stops, laps, tyres, react to safety cars) and
#        simulate it a few hundred times against the real strategy
#     3. see what the AI strategist would have done in the same race
#     4. read a season "strategy report card" per team
# What it reads: app_data/races.pkl (the 2024 and 2025 races, made by
#   final_test.py), the trained models in models/ and models/final/, and
#   results/strategist_2024.parquet and results/strategist_2025.parquet.
# What it produces: nothing on disk; it draws the pages.
# Which files use it: none. It runs on a small free server with no GPU.
#
# Run:  .venv\Scripts\python.exe -m streamlit run app.py
#
# Each season is simulated with models that never saw it: 2024 with models
# trained on 2022 to 2023, and 2025 with models trained on 2022 to 2024.

import pickle
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st
import torch

import chart_style as style
import simulator
import strategy_search
from simulator import LapModel, RaceSet
from strategy_env import StrategyEnv
from train_agent import drive, load_strategist
from train_network import COMPOUND_NAMES

RACES_FILE = Path("app_data/races.pkl")
MODELS = {  # season -> the folder of models that never saw that season
    2024: Path("models"),
    2025: Path("models/final"),
}
SIMULATIONS = 300
SEED = 7
STATUS_NAMES = {1: "VSC", 2: "Safety car"}


@st.cache_resource(show_spinner="Loading the models...")
def load_season(year: int):
    """Everything needed to simulate one season, loaded once and kept."""
    folder = MODELS[year]
    races = [race for race in pickle.loads(RACES_FILE.read_bytes()) if race.year == year]
    lap_model = LapModel(folder / "lap_time_network.pt")
    stats = simulator.load_stats(folder / "race_stats.json")
    race_set = RaceSet(races, lap_model, stats)
    strategist = load_strategist(folder / "strategist.pt", race_set.device)
    return race_set, lap_model, stats, strategist


def make_plan(stops: list, max_laps: int) -> torch.Tensor:
    """Turn [(lap, compound number), ...] into a plan the simulator reads."""
    plan = torch.zeros(max_laps, dtype=torch.long)
    for lap, compound in stops:
        plan[lap - 1] = 1 + compound
    return plan


@st.cache_data(show_spinner="Simulating...")
def simulate(year: int, race: int, car: int, stops: tuple, react: bool, real_events: bool):
    """Simulate the real strategy, the edited one and the AI strategist in
    the same random races. Returns finishing positions and race times."""
    race_set, lap_model, stats, strategist = load_season(year)
    info = race_set.races[race]

    def run(plan, fixed, ages=None):
        positions, times = strategy_search.simulate_plans(
            race_set, lap_model, stats, race, car, plan[None], SIMULATIONS, real_events,
            seed=SEED, fixed=fixed, ages=ages)
        return positions[0].cpu().numpy(), times[0].cpu().numpy()

    real = run(race_set.plan[race, car].cpu(), fixed=real_events,
               ages=race_set.plan_age[race, car].cpu()[None])
    yours = run(make_plan(list(stops), race_set.max_laps), fixed=not react)

    env = StrategyEnv(race_set, lap_model, stats, real_events=real_events)
    index = torch.full((SIMULATIONS,), race, device=race_set.device)
    agent = torch.full((SIMULATIONS,), car, device=race_set.device)
    group = torch.arange(SIMULATIONS, device=race_set.device)
    driven = drive(env, strategist, index, agent, seed=SEED, group=group)
    actions = driven["actions"][:, :info.total_laps].cpu().numpy()
    ai = (driven["position"].cpu().numpy(), driven["time"].cpu().numpy())
    return {"real": real, "yours": yours, "ai": ai, "ai_actions": actions,
            "status": driven["status"][:, :info.total_laps].cpu().numpy()}


def stints(start_compound: int, plan) -> list:
    """[(first lap, last lap, compound number), ...] for a plan."""
    plan = np.asarray(plan)
    result, first, compound = [], 1, start_compound
    for lap in np.nonzero(plan)[0] + 1:
        result.append((first, lap, compound))
        first, compound = lap + 1, int(plan[lap - 1]) - 1
    result.append((first, len(plan), compound))
    return result


def timeline_chart(rows: list, total_laps: int, status):
    """Horizontal bars, one per strategy, coloured by tyre compound, with
    safety car and VSC laps shaded behind them."""
    figure, axes = style.new_figure(width=9, height=0.75 * len(rows) + 1.3)
    for lap in np.nonzero(np.asarray(status))[0] + 1:
        axes.axvspan(lap - 0.5, lap + 0.5, color=style.GRID, zorder=0)
    for y, (label, start_compound, plan) in enumerate(rows):
        for first, last, compound in stints(start_compound, plan[:total_laps]):
            name = COMPOUND_NAMES[compound]
            axes.barh(y, last - first + 1, left=first - 0.5, height=0.5,
                      color=style.COMPOUND_COLOURS[name], edgecolor=style.SURFACE,
                      linewidth=2, zorder=2)
            if last - first >= 5:
                axes.text((first + last) / 2, y, f"{name.title()} ({last - first + 1})",
                          ha="center", va="center", fontsize=8, color=style.INK, zorder=3)
    axes.set_yticks(range(len(rows)), [row[0] for row in rows])
    axes.invert_yaxis()
    axes.set_xlim(0.5, total_laps + 0.5)
    axes.set_xlabel("Lap  (shaded laps: safety car or VSC)")
    style.tidy(axes, "")
    axes.grid(False)
    figure.tight_layout()
    return figure


def position_chart(results: dict, names: dict):
    """How often each strategy finished in each position: three small
    charts sharing one axis, so they can be compared at a glance."""
    colours = {"real": style.BASELINE, "yours": style.ORANGE, "ai": style.BLUE}
    figure, axes = style.new_figure(width=9, height=5.2)
    figure.clear()
    panels = figure.subplots(3, 1, sharex=True, sharey=True)
    for panel, key in zip(panels, ["real", "yours", "ai"]):
        positions = results[key][0]
        share = np.bincount(positions.astype(int), minlength=21)[1:] / len(positions)
        panel.bar(np.arange(1, 21), share * 100, color=colours[key], width=0.8)
        style.tidy(panel, f"{names[key]}: average position {positions.mean():.1f}", "y")
        panel.set_ylabel("% of races")
    panels[-1].set_xticks(range(1, 21))
    panels[-1].set_xlabel("Finishing position")
    figure.tight_layout()
    return figure


def show_real_race(info, car: int):
    st.subheader("1. What really happened")
    left, middle, right = st.columns(3)
    left.metric("Started", f"P{int(info.grid[car])}")
    middle.metric("Finished", f"P{int(info.finish_position[car])}")
    right.metric("Stops", int((info.plan[car] > 0).sum()))
    cautions = [f"lap {lap + 1} ({STATUS_NAMES[code]})"
                for lap, code in enumerate(info.status) if code > 0]
    st.caption("Safety car and VSC laps: " + (", ".join(cautions) if cautions else "none"))
    st.pyplot(timeline_chart(
        [("Real strategy", int(info.start_compound[car]), info.plan[car])],
        info.total_laps, info.status))


def strategy_editor(info, car: int) -> tuple:
    """The controls for editing a strategy. Returns (stops, react, legal)."""
    st.subheader("2. Try a different strategy")
    start = int(info.start_compound[car])
    st.caption(f"The car starts on {COMPOUND_NAMES[start].title()} tyres, as it really did.")
    stop_count = st.radio("Number of stops", [1, 2], horizontal=True, key="stops")
    real_laps = (np.nonzero(info.plan[car])[0] + 1).tolist()
    stops, columns = [], st.columns(stop_count)
    for i, column in enumerate(columns):
        default = real_laps[i] if i < len(real_laps) else info.total_laps * (i + 1) // 3
        default = int(np.clip(default, 2 + i, info.total_laps - 2))
        lap = column.slider(f"Stop {i + 1}: pit at the end of lap", 1, info.total_laps - 1,
                            default, key=f"lap{i}")
        compound = column.selectbox(f"Stop {i + 1}: fit", COMPOUND_NAMES,
                                    index=(start + 1 + i) % 3, key=f"tyre{i}",
                                    format_func=str.title)
        stops.append((lap, COMPOUND_NAMES.index(compound)))
    react = st.checkbox(
        "Pit under a safety car: if one appears within 8 laps before a planned stop"
        " (4 for a VSC), stop straight away instead", value=True)
    laps = [lap for lap, _ in stops]
    used = {start} | {compound for _, compound in stops}
    legal = True
    if len(set(laps)) < len(laps) or laps != sorted(laps):
        st.warning("The stops must be on different laps, in order.")
        legal = False
    elif len(used) < 2:
        st.warning("The rules require at least two different tyre compounds in a race.")
        legal = False
    return tuple(stops), react, legal


def show_results(info, car: int, results: dict, stops: tuple):
    names = {"real": "The team's real strategy", "yours": "Your strategy",
             "ai": "The AI strategist"}
    real_position, real_time = results["real"][0].mean(), results["real"][1].mean()
    columns = st.columns(3)
    for column, key in zip(columns, ["real", "yours", "ai"]):
        position, seconds = results[key][0].mean(), real_time - results[key][1].mean()
        column.metric(
            names[key], f"P{position:.1f}",
            None if key == "real" else f"{real_position - position:+.1f} places,"
                                       f" {seconds:+.1f}s",
        )
    st.caption(f"Average over {SIMULATIONS} simulated versions of this race. Positive"
               " numbers are places and seconds gained over the team's real strategy.")

    st.subheader("3. What the AI strategist did")
    actions = results["ai_actions"]
    stop_counts = (actions > 0).sum(axis=1)
    st.write(
        f"It was given this car and decided lap by lap. Across {SIMULATIONS} simulations it"
        f" made {stop_counts.mean():.1f} stops on average"
        f" ({(stop_counts == 1).mean():.0%} one stop, {(stop_counts == 2).mean():.0%} two)."
        " One of its races is shown below, next to the others.")
    start = int(info.start_compound[car])
    your_plan = make_plan(list(stops), info.total_laps).numpy()
    st.pyplot(timeline_chart(
        [("Real strategy", start, info.plan[car]), ("Your strategy", start, your_plan),
         ("AI strategist\n(one example)", start, actions[0])],
        info.total_laps, results["status"][0]))
    st.pyplot(position_chart(results, names))


def report_card(year: int):
    st.subheader(f"4. Strategy report card, {year}")
    file = Path(f"results/strategist_{year}.parquet")
    if not file.exists():
        st.info("The report card for this season has not been produced yet.")
        return
    table = strategy_search.fair_comparisons(pl.read_parquet(file))
    card = table.group_by("Team").agg(
        pl.len().alias("Driver races"),
        (pl.col("RealPosition") - pl.col("FixedPosition")).mean().round(2)
        .alias("Places a fixed plan would gain"),
        (pl.col("RealPosition") - pl.col("AgentPosition")).mean().round(2)
        .alias("Places the AI strategist would gain"),
        pl.col("AgentSeconds").mean().round(1).alias("Seconds the AI strategist would gain"),
    ).sort("Places the AI strategist would gain")
    st.write(
        "For every dry race, each driver's real strategy was simulated against the best"
        " fixed plan found by brute force and against the AI strategist. **Smaller numbers"
        " are better for the team**: they mean the simulator found little to improve."
        " Teams are listed best first.")
    st.dataframe(card, hide_index=True, use_container_width=True)
    st.caption(
        "Read this with care. These are gains inside a simulator that has measured"
        " flaws (see the README), it does not know about tyre sets available, damage,"
        " team orders or what the team was protecting against. A large number means"
        " 'worth a closer look', not 'the team got it wrong'.")


def main():
    st.set_page_config(page_title="F1START", layout="wide")
    st.title("F1START: what if they had pitted differently?")
    st.caption("An F1 race strategy simulator with an AI strategist, built from public"
               " timing data. Dry races only. Not associated with Formula 1 or any team.")
    if not RACES_FILE.exists():
        st.error(f"{RACES_FILE} is missing. Run final_test.py first.")
        return

    year = st.sidebar.selectbox("Season", sorted(MODELS, reverse=True))
    race_set, _, _, _ = load_season(year)
    labels = [race.name for race in race_set.races]
    race = st.sidebar.selectbox("Race", range(len(labels)), format_func=labels.__getitem__)
    info = race_set.races[race]
    cars = strategy_search.finishers(info)
    car = st.sidebar.selectbox(
        "Driver", cars, format_func=lambda c: f"{info.drivers[c]} ({info.teams[c]})")
    conditions = st.sidebar.radio(
        "Safety cars", ["As they really happened", "Random, as before the race"])
    st.sidebar.caption(
        f"Simulated on: {race_set.device}. Models for {year} were trained only on"
        f" seasons before {year}.")

    show_real_race(info, car)
    stops, react, legal = strategy_editor(info, car)
    if legal:
        results = simulate(year, race, car, stops, react,
                           conditions == "As they really happened")
        show_results(info, car, results, stops)
    st.divider()
    report_card(year)


if __name__ == "__main__":
    main()
