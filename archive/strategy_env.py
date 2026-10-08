# strategy_env.py
#
# What it does: the race as a game for a strategist to play (Phase 2). It
#   wraps the simulator in the standard reinforcement learning interface
#   (the "Gymnasium" style): reset() starts races and returns what the
#   strategist can see, step(action) drives one lap and returns the new view
#   and a reward. The strategist controls ONE car per race; every other car
#   follows its real strategy. Thousands of races run side by side on the GPU.
# What it reads: nothing itself; it is given a RaceSet, LapModel and stats
#   from simulator.py.
# What it produces: nothing on disk.
# Which files use it: train_agent.py (to train the AI strategist),
#   evaluate_strategist.py and app.py (to let the trained strategist drive).
#
# The actions, once per lap: 0 stay out, 1 pit for soft, 2 pit for medium,
# 3 pit for hard. A stop happens at the end of the lap it is chosen on.

import gymnasium
import numpy as np
import torch

import simulator
import strategy_search
from simulator import GAP_CAP_SECONDS, NEVER, RaceSim

# --- The reward (docs/DECISIONS.md, number 27) ---
# Each lap: minus the time lost to the typical car on that lap, in units of
# this many seconds. A pit stop of 22s therefore costs about 2.2 at once,
# and fresh tyres pay it back a little every lap.
REWARD_SECONDS = 10.0
# At the finish: minus this for every finishing place (1st loses the least).
REWARD_PER_PLACE = 0.2

OBSERVATION_NAMES = [
    "race fraction done", "laps remaining", "tyre age", "on soft", "on medium", "on hard",
    "position", "gap ahead", "gap behind", "green", "vsc", "safety car", "stops made",
    "used soft", "used medium", "used hard", "pit loss", "qualifying gap",
    "pace now", "pace on new soft", "pace on new medium", "pace on new hard",
    "laps past tyre limit",
    # Which compounds it is allowed to fit. These three MUST stay last: the
    # strategist reads them to rule out the compounds it may not choose.
    "may fit soft", "may fit medium", "may fit hard",
]


class StrategyEnv:
    """B races at once; the strategist drives one car in each."""

    def __init__(self, race_set, lap_model, stats, real_events: bool = False):
        self.set, self.model, self.stats = race_set, lap_model, stats
        self.real_events = real_events
        self.device = race_set.device
        self.observation_space = gymnasium.spaces.Box(
            -np.inf, np.inf, (len(OBSERVATION_NAMES),), dtype=np.float32)
        self.action_space = gymnasium.spaces.Discrete(4)
        # Which cars may be driven: the ones that really finished the race.
        finisher = torch.zeros(len(race_set.races), simulator.CARS, device=self.device)
        for r, race in enumerate(race_set.races):
            ok = (race.retire_lap == NEVER) & (race.finish_position < 99)
            finisher[r, :len(ok)] = torch.tensor(ok, dtype=torch.float32)
        self.finisher = finisher
        # The compounds each car's team really used in each race. (R, D, 3)
        team_tyres = torch.ones(len(race_set.races), simulator.CARS, 3, dtype=torch.bool,
                                device=self.device)
        for r, race in enumerate(race_set.races):
            for car in range(len(race.drivers)):
                team_tyres[r, car] = False
                team_tyres[r, car, list(strategy_search.team_compounds(race, car))] = True
        self.team_tyres = team_tyres

    def random_episodes(self, count: int, generator) -> tuple:
        """Pick `count` random races, a random finisher in each, and which
        compounds it may fit.

        In half the races the strategist is limited to the compounds that
        car's team really used, in the other half it may fit anything. It has
        to be trained on both, because it is evaluated on both: a strategist
        that only ever practised with all three compounds has no plan for
        the two compound rule when one of them is taken away
        (docs/DECISIONS.md, number 33).
        """
        race = torch.randint(len(self.set.races), (count,), generator=generator,
                             device=self.device)
        agent = torch.multinomial(self.finisher[race], 1, generator=generator)[:, 0]
        limited = torch.rand(count, generator=generator, device=self.device) < 0.5
        allowed = self.team_tyres[race, agent] | ~limited[:, None]
        return race, agent, allowed

    def reset(self, race, agent, seed: int = 0, group=None, allowed=None):
        """Start the races. `race` (B,) and `agent` (B,) say which race each
        one is and which car the strategist drives. `allowed` (B, 3) can
        limit which compounds may be fitted. Returns (view, info)."""
        self.agent = agent
        self.allowed = (torch.ones(len(race), 3, dtype=torch.bool, device=self.device)
                        if allowed is None else allowed)
        plan = self.set.plan[race].clone()
        rows = torch.arange(len(race), device=self.device)
        plan[rows, agent] = 0  # the strategist's car has no plan: it decides
        status = None
        if not self.real_events:
            generator = torch.Generator(device=self.device).manual_seed(seed + 1)
            status = simulator.sample_status(self.set, race, self.stats, generator)
            if group is not None:  # races in a group share their safety cars
                first_of_group = torch.zeros(int(group.max()) + 1, dtype=torch.long,
                                             device=self.device)
                first_of_group[group.flip(0)] = rows.flip(0)
                status = status[first_of_group[group]]
        self.sim = RaceSim(self.set, self.model, self.stats, race, status=status,
                           plan=plan, group=group, seed=seed)
        self.rows = rows
        self.forced_stops = torch.zeros(len(race), dtype=torch.long, device=self.device)
        return self.observe(), {}

    def legal_action(self, action):
        """Enforce the two compound rule.

        On the last lap where a stop is still possible, a car that has only
        used one compound is made to pit for a different one. In the real
        sport breaking this rule means disqualification; forcing the stop
        keeps every simulated race legal, and the strategist learns that
        leaving it this late is a very expensive way to do it.
        """
        sim = self.sim
        used = sim.used[self.rows, self.agent]  # (B, 3)
        last_chance = sim.lap == sim.total_laps - 1
        choice_is_new = (action > 0) & ~used.gather(1, (action - 1).clamp(min=0)[:, None])[:, 0]
        must_force = last_chance & (used.sum(dim=1) < 2) & ~choice_is_new
        # Prefer medium, then hard, then soft among the unused (and allowed)
        # compounds.
        preference = torch.tensor([1, 2, 0], device=self.device)
        unused_first = (~used & self.allowed)[:, preference].float().argmax(dim=1)
        forced = 1 + preference[unused_first]
        self.forced_stops += must_force.long()
        return torch.where(must_force, forced, action)

    def step(self, action):
        """Drive one lap. Returns (view, reward, terminated, truncated, info)."""
        sim = self.sim
        lap = sim.lap
        racing = lap <= sim.total_laps  # races that are not over yet
        action = self.legal_action(action)
        sim.step(self.agent, action)

        # Time lost to the typical car this lap. Comparing with the field,
        # not with the clock, removes things the strategist cannot control:
        # a safety car slows everyone, so it costs no reward.
        lap_times = torch.where(sim.running(lap), sim.last_lap_time, torch.nan)
        typical = lap_times.nanmedian(dim=1).values
        own = sim.last_lap_time[self.rows, self.agent]
        reward = -(own - typical) / REWARD_SECONDS
        finishing = lap == sim.total_laps
        position = sim.final_positions()[self.rows, self.agent]
        reward = reward - torch.where(finishing, REWARD_PER_PLACE * position, 0.0)
        reward = torch.where(racing, reward, 0.0)

        terminated = sim.lap > sim.total_laps
        truncated = torch.zeros_like(terminated)
        # The action that was really carried out (0 once the race is over, or
        # on the final lap, where stops are not allowed).
        pitted = sim.last_pitted[self.rows, self.agent]
        info = {"position": position, "action": torch.where(pitted, action, 0)}
        return self.observe(), reward, terminated, truncated, info

    def observe(self) -> torch.Tensor:
        """What the strategist can see before choosing: (B, 26) numbers,
        all scaled to be roughly between -1 and 1. See OBSERVATION_NAMES."""
        sim, rows, agent = self.sim, self.rows, self.agent
        lap = sim.lap
        mine = lambda tensor: tensor[rows, agent]
        running = sim.running(lap)
        gap_ahead, _ = sim.gap_ahead(running)

        time = torch.where(running, sim.time, torch.inf)
        ordered, order = time.sort(dim=1)
        behind = torch.full_like(ordered, GAP_CAP_SECONDS)
        behind[:, :-1] = (ordered[:, 1:] - ordered[:, :-1]).nan_to_num(GAP_CAP_SECONDS)
        gap_behind = torch.empty_like(behind).scatter_(1, order, behind.clamp(0, GAP_CAP_SECONDS))
        if lap == 1:
            gap_behind = torch.full_like(gap_behind, GAP_CAP_SECONDS)

        status = sim.status[:, min(lap, sim.status.shape[1]) - 1]
        compound = mine(sim.compound)
        tyre_age = mine(sim.tyre_age)
        # What the lap time model expects: the pace on the current tyres,
        # and on a new set of each compound in clear air. This is the same
        # kind of tyre model a real strategist has on the pit wall.
        open_road = torch.full_like(gap_ahead, GAP_CAP_SECONDS)
        pace_now = mine(sim.predicted_pace(sim.tyre_age, sim.compound, gap_ahead)[0])
        new = [mine(sim.predicted_pace(torch.zeros_like(sim.tyre_age),
                                       torch.full_like(sim.compound, c), open_road)[0])
               for c in range(3)]
        past_limit = (tyre_age + 1 - sim.tyre_limit[compound]).clamp(min=0)

        total = sim.total_laps.float()
        parts = [
            (lap - 1) / total, (total - lap) / 70, tyre_age / 40,
            *(compound == c for c in range(3)),
            mine(sim.positions()) / 20, mine(gap_ahead) / 10, mine(gap_behind) / 10,
            *(status == s for s in range(3)),
            mine(sim.stops) / 3,
            *(mine(sim.used)[:, c] for c in range(3)),
            sim.pit_loss[:, 0] / 30, mine(sim.quali_gap) / 3,
            pace_now / 3, *(pace / 3 for pace in new),
            past_limit / 10,
            *(self.allowed[:, c] for c in range(3)),
        ]
        return torch.stack([part.float() for part in parts], dim=1)
