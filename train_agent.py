# train_agent.py
#
# What it does: trains the AI strategist (Phase 3) with reinforcement
#   learning, using the PPO algorithm, on the GPU. The strategist plays
#   thousands of simulated races at once in strategy_env.py, each lap
#   choosing to stay out or pit for a compound, and gradually learns which
#   choices lead to less time lost and better finishing positions.
# What it reads: data/laps.parquet, models/lap_time_network.pt and
#   models/race_stats.json (through simulator.py). It trains on the 2022 to
#   2023 races only, with random safety cars.
# What it produces: models/strategist.pt and a training log on the screen.
# Which files use it: evaluate_strategist.py and app.py load the strategist
#   with load_strategist() and let it drive with drive(). final_test.py
#   calls train() again with 2024 included.
#
# Run:  .venv\Scripts\python.exe train_agent.py
#
# PPO in plain English. The strategist is a network with two heads:
#   - the POLICY: for each possible action, how likely it is to choose it
#   - the VALUE: how well it expects the rest of the race to go from here
# Each round it plays a few thousand races using its current policy, with
# some randomness so that it tries things. Afterwards every decision is
# scored: did the race go better or worse from there than the value head
# expected? Decisions that turned out better than expected are made more
# likely, the others less likely. "Proximal" means each round is only
# allowed to change the policy a little, which keeps learning stable.

import time
from pathlib import Path

import torch
from torch import nn

import simulator
from simulator import LapModel, RaceSet
from strategy_env import OBSERVATION_NAMES, StrategyEnv

STRATEGIST_FILE = Path("models/strategist.pt")
DEVICE = simulator.DEVICE

# Settings (docs/DECISIONS.md, number 28).
RACES_PER_ROUND = 4096  # simulated races played side by side each round
ROUNDS = 150
LEARNING_RATE = 0.0003
PASSES_PER_ROUND = 4  # how many times each round's decisions are learned from
BATCH_SIZE = 32768
CLIP = 0.2  # the "proximal" part: how far the policy may move per round
LAMBDA = 0.95  # how far ahead credit for a decision reaches
VALUE_WEIGHT = 0.5
EXPLORE_WEIGHT = 0.003  # a small reward for keeping its options open
SEED = 0
# A brand new policy starts out choosing "stay out" about 98% of the time.
# Without this it would pit on three laps out of four, every race would be a
# disaster, and it would take a long time to discover that stopping rarely
# is the place to start.
STAY_OUT_HEAD_START = 5.0


class Strategist(nn.Module):
    """The policy and value networks."""

    def __init__(self, observation_size: int = len(OBSERVATION_NAMES), width: int = 128):
        super().__init__()

        def body(outputs):
            return nn.Sequential(
                nn.Linear(observation_size, width), nn.Tanh(),
                nn.Linear(width, width), nn.Tanh(),
                nn.Linear(width, outputs),
            )

        self.policy = body(4)  # one score per action
        self.value = body(1)
        with torch.no_grad():
            self.policy[-1].weight *= 0.01
            self.policy[-1].bias.zero_()
            self.policy[-1].bias[0] = STAY_OUT_HEAD_START

    def forward(self, view):
        """Returns (the choice distribution over actions, the value)."""
        choices = torch.distributions.Categorical(logits=self.policy(view))
        return choices, self.value(view)[:, 0]


def play_round(env: StrategyEnv, strategist: Strategist, generator, seed: int) -> dict:
    """Play RACES_PER_ROUND random races; remember everything that happened."""
    race, agent = env.random_episodes(RACES_PER_ROUND, generator)
    view, _ = env.reset(race, agent, seed=seed)
    memory = {name: [] for name in ["view", "action", "chance", "value", "reward", "racing"]}
    for _ in range(env.set.max_laps):
        racing = ~env.sim.finished()
        with torch.no_grad():
            choices, value = strategist(view)
            action = choices.sample()
        next_view, reward, _, _, info = env.step(action)
        for name, item in [("view", view), ("action", action), ("value", value),
                           ("chance", choices.log_prob(action)), ("reward", reward),
                           ("racing", racing)]:
            memory[name].append(item)
        view = next_view
    memory = {name: torch.stack(items) for name, items in memory.items()}  # (laps, B, ...)
    memory["position"] = info["position"].float()
    memory["stops"] = env.sim.stops[env.rows, env.agent].float()
    memory["forced"] = env.forced_stops.float()
    return memory


def score_decisions(memory: dict) -> tuple:
    """For every decision: how much better did things go than expected?

    Working backwards from the finish. `advantage` is "what happened after
    this decision" minus "what the value head expected". LAMBDA blends
    short and long horizons (this is called GAE). Returns (advantage, the
    total reward that actually followed each decision).
    """
    reward, value, racing = memory["reward"], memory["value"], memory["racing"].float()
    advantage = torch.zeros_like(reward)
    running = torch.zeros_like(reward[0])
    next_value = torch.zeros_like(value[0])
    for lap in reversed(range(len(reward))):
        surprise = reward[lap] + next_value - value[lap]
        running = (surprise + LAMBDA * running) * racing[lap]
        advantage[lap] = running
        next_value = value[lap] * racing[lap]
    return advantage, advantage + value


def learn(strategist: Strategist, optimizer, memory: dict):
    """The PPO update: nudge the policy towards decisions that beat
    expectations, and the value head towards what really happened."""
    advantage, outcome = score_decisions(memory)
    keep = memory["racing"].reshape(-1)  # ignore laps after a race finished
    flat = lambda tensor: tensor.reshape(-1, *tensor.shape[2:])[keep]
    view, action, old_chance = flat(memory["view"]), flat(memory["action"]), flat(memory["chance"])
    advantage, outcome = flat(advantage), flat(outcome)
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    for _ in range(PASSES_PER_ROUND):
        order = torch.randperm(len(view), device=view.device)
        for start in range(0, len(view), BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE]
            choices, value = strategist(view[rows])
            # How much more (or less) likely the new policy makes each old
            # decision. Clipping it stops one round changing the policy a lot.
            ratio = torch.exp(choices.log_prob(action[rows]) - old_chance[rows])
            gain = torch.minimum(ratio * advantage[rows],
                                 ratio.clamp(1 - CLIP, 1 + CLIP) * advantage[rows])
            value_error = (value - outcome[rows]) ** 2
            loss = (-gain.mean() + VALUE_WEIGHT * value_error.mean()
                    - EXPLORE_WEIGHT * choices.entropy().mean())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(strategist.parameters(), 0.5)
            optimizer.step()


def train(env: StrategyEnv, rounds: int = ROUNDS) -> Strategist:
    torch.manual_seed(SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    strategist = Strategist().to(DEVICE)
    optimizer = torch.optim.Adam(strategist.parameters(), lr=LEARNING_RATE)
    started = time.perf_counter()
    print(f"{'round':>5} {'reward':>8} {'position':>9} {'stops':>6} {'forced':>7} {'seconds':>8}")
    for round_number in range(1, rounds + 1):
        memory = play_round(env, strategist, generator, seed=round_number)
        learn(strategist, optimizer, memory)
        if round_number == 1 or round_number % 10 == 0:
            print(f"{round_number:>5} {memory['reward'].sum(dim=0).mean().item():>8.2f}"
                  f" {memory['position'].mean().item():>9.2f}"
                  f" {memory['stops'].mean().item():>6.2f}"
                  f" {memory['forced'].mean().item():>7.1%}"
                  f" {time.perf_counter() - started:>8.0f}")
    return strategist


def save_strategist(strategist: Strategist, file: Path = STRATEGIST_FILE):
    file.parent.mkdir(exist_ok=True)
    torch.save({"weights": strategist.state_dict(), "observations": OBSERVATION_NAMES}, file)
    print(f"\nSaved the strategist to {file}")


def load_strategist(file: Path = STRATEGIST_FILE, device: str = DEVICE) -> Strategist:
    strategist = Strategist().to(device)
    strategist.load_state_dict(torch.load(file, map_location=device)["weights"])
    return strategist.eval()


@torch.no_grad()
def drive(env: StrategyEnv, strategist: Strategist, race, agent, seed: int, group=None,
          action_seed: int = 0, allowed=None) -> dict:
    """Let the trained strategist drive the given races to the finish.

    `allowed` (B, 3) can limit which compounds it may fit: choices for other
    compounds are removed before it picks.

    Its choices are sampled from the policy, as in training. Taking only its
    single most likely action each lap would not work: when it wants to stop
    "some time in the next few laps", staying out is still the most likely
    choice on each individual lap, so it would never stop at all.
    Returns final positions and times (B,), and the action of every lap.
    """
    generator = torch.Generator(device=env.device).manual_seed(action_seed)
    view, _ = env.reset(race, agent, seed=seed, group=group, allowed=allowed)
    actions = []
    for _ in range(env.set.max_laps):
        chances = strategist(view)[0].probs
        chances[:, 1:] = chances[:, 1:] * env.allowed
        action = torch.multinomial(chances, 1, generator=generator)[:, 0]
        view, _, _, _, info = env.step(action)
        actions.append(info["action"])
    return {
        "position": info["position"].float(),
        "time": env.sim.time[env.rows, env.agent],
        "actions": torch.stack(actions, dim=1),  # (B, laps)
        "forced": env.forced_stops,
        "status": env.sim.status,
    }


def main():
    lap_model = LapModel()
    stats = simulator.load_stats()
    race_set = RaceSet(simulator.load_races(["train"]), lap_model, stats)
    env = StrategyEnv(race_set, lap_model, stats, real_events=False)
    print(f"Training the strategist on {len(race_set.races)} dry 2022 to 2023 races,"
          f" {RACES_PER_ROUND} simulated races per round, on {DEVICE}.\n")
    strategist = train(env)
    save_strategist(strategist)


if __name__ == "__main__":
    main()
