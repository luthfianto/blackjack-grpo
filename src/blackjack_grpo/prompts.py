import json
from .game import State, action_values

RULES = (
    "Blackjack rules: infinite deck; only HIT or STAND; an ace counts as 11 when possible, otherwise 1. "
    "The dealer stands on all totals of 17 or higher, including soft 17. "
    "A win pays +1, a draw 0, and a loss -1; there is no natural-blackjack bonus. "
    "Choose the action with the best expected return. Explain in one or two short sentences. "
    'Return only a JSON object with exactly two fields: {"reasoning":"your explanation","action":"HIT or STAND"}. '
    "Do not use Markdown or an extended thinking trace."
)


def prompt(state, wording="train"):
    ace = "yes" if state.usable_ace else "no"
    dealer = "ace (1)" if state.dealer == 1 else str(state.dealer)
    if wording == "train":
        observation = f"Player total: {state.total}. Dealer upcard: {dealer}. Usable ace: {ace}. What is your move?"
    elif wording == "heldout":
        observation = (f"Decide my next move. I hold a {'soft' if state.usable_ace else 'hard'} "
                       f"{state.total}; the dealer shows {dealer}. A usable ace is {'present' if state.usable_ace else 'absent'}.")
    else:
        raise ValueError(f"Unknown wording: {wording}")
    return [{"role": "system", "content": RULES}, {"role": "user", "content": observation}]


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def parse_completion(text):
    try:
        result = json.loads(text, object_pairs_hook=_unique_keys)
        if not isinstance(result, dict) or set(result) != {"reasoning", "action"}:
            return None
        if not isinstance(result["reasoning"], str) or not result["reasoning"].strip():
            return None
        if result["action"] not in ("HIT", "STAND"):
            return None
        return result
    except (ValueError, TypeError):
        return None


def reward(text, state):
    parsed = parse_completion(text)
    return -2.0 if parsed is None else action_values(state)[parsed["action"]]


def blackjack_reward(completions, total, dealer, usable_ace, **kwargs):
    """TRL supplies each dataset observation repeated for its completion group."""
    scores = []
    for completion, player, upcard, soft in zip(completions, total, dealer, usable_ace, strict=True):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        scores.append(reward(text, State(int(player), int(upcard), bool(soft))))
    return scores
