"""Exact infinite-deck Blackjack-v1 solver (natural=False, sab=False).

An action value assumes optimal decisions *after* that action. It is not
necessarily the return achieved by the language model's complete policy.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

CARDS = tuple((card, (4 if card == 10 else 1) / 13) for card in range(1, 11))
ACTIONS = ("STAND", "HIT")


@dataclass(frozen=True, order=True)
class State:
    total: int
    dealer: int
    usable_ace: bool

    def __post_init__(self):
        if not (1 <= self.dealer <= 10 and 4 <= self.total <= 21):
            raise ValueError("Expected total 4..21 and dealer upcard 1..10")
        if self.usable_ace and self.total < 12:
            raise ValueError("A reachable soft hand has total 12..21")

    @property
    def key(self):
        return f"{self.total}:{self.dealer}:{int(self.usable_ace)}"


def states():
    return [State(total, dealer, soft) for soft in (False, True)
            for total in range(12 if soft else 4, 22) for dealer in range(1, 11)]


def draw(total, usable_ace, card):
    """Update a hand after drawing a card; totals >21 indicate a bust."""
    if card == 1 and total + 11 <= 21:
        total += 11
        usable_ace = True
    else:
        total += card
    if total > 21 and usable_ace:
        total -= 10
        usable_ace = False
    return total, usable_ace


@lru_cache(None)
def dealer_outcomes(total, usable_ace):
    """Probability vector: dealer 17,18,19,20,21,bust; stand on soft 17."""
    if total >= 17:
        result = [0.0] * 6
        result[5 if total > 21 else total - 17] = 1.0
        return tuple(result)
    result = [0.0] * 6
    for card, probability in CARDS:
        child = dealer_outcomes(*draw(total, usable_ace, card))
        for i, value in enumerate(child):
            result[i] += probability * value
    return tuple(result)


@lru_cache(None)
def dealer_distribution(upcard):
    # Average over the hidden card, without conditioning on a dealer peek.
    return dealer_outcomes(11 if upcard == 1 else upcard, upcard == 1)


@lru_cache(None)
def action_values(state):
    probabilities = dealer_distribution(state.dealer)
    stand = probabilities[-1]
    for dealer_total, probability in zip(range(17, 22), probabilities[:-1]):
        stand += probability * ((state.total > dealer_total) - (state.total < dealer_total))
    hit = 0.0
    for card, probability in CARDS:
        total, soft = draw(state.total, state.usable_ace, card)
        value = -1.0 if total > 21 else max(action_values(State(total, state.dealer, soft)).values())
        hit += probability * value
    return {"STAND": stand, "HIT": hit}


def optimal_action(state):
    values = action_values(state)
    return max(ACTIONS, key=values.__getitem__)


def simulate(policy, episodes=100_000, seed=2026):
    """Independent Gymnasium simulations; returns episode-return uncertainty."""
    import gymnasium as gym

    env = gym.make("Blackjack-v1", natural=False, sab=False)
    total = squared = 0.0
    wins = draws = losses = 0
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed if episode == 0 else None)
            done = False
            while not done:
                state = State(int(observation[0]), int(observation[1]), bool(observation[2]))
                action = policy(state)
                if action not in ACTIONS:
                    raise ValueError(f"Invalid policy action: {action}")
                observation, reward, terminated, truncated, _ = env.step(int(action == "HIT"))
                done = terminated or truncated
            total += reward
            squared += reward * reward
            wins += reward > 0
            draws += reward == 0
            losses += reward < 0
    finally:
        env.close()
    mean = total / episodes
    variance = (squared - episodes * mean * mean) / (episodes - 1) if episodes > 1 else 0.0
    error = math.sqrt(max(0.0, variance) / episodes)
    return dict(episodes=episodes, seed=seed, mean_return=mean, standard_error=error,
                ci95=[mean - 1.96 * error, mean + 1.96 * error], wins=wins, draws=draws, losses=losses)
