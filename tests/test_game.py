import itertools
import unittest

from blackjack_grpo.game import State, action_values, dealer_distribution, dealer_outcomes, draw, optimal_action, states
from blackjack_grpo.prompts import blackjack_reward, parse_completion, prompt, reward


class BlackjackTests(unittest.TestCase):
    def test_reachable_observations(self):
        import gymnasium.envs.toy_text.blackjack as bj
        reachable = set()
        for size in (2, 3):
            for hand in itertools.product(range(1, 11), repeat=size):
                total = bj.sum_hand(hand)
                if total <= 21:
                    for dealer in range(1, 11):
                        reachable.add(State(total, dealer, bj.usable_ace(hand)))
        self.assertEqual(reachable, set(states()))
        self.assertEqual(len(states()), 280)

    def test_transitions_match_gymnasium(self):
        import gymnasium.envs.toy_text.blackjack as bj
        for hand in itertools.product(range(1, 11), repeat=2):
            if bj.sum_hand(hand) <= 21:
                for card in range(1, 11):
                    actual = draw(bj.sum_hand(hand), bj.usable_ace(hand), card)
                    expected = bj.sum_hand((*hand, card)), bj.usable_ace((*hand, card))
                    self.assertEqual(actual, expected)
        self.assertEqual(draw(20, False, 10), (30, False))
        self.assertEqual(draw(18, True, 9), (17, False))
        self.assertEqual(draw(12, True, 1), (13, True))

    def test_dealer_rules_and_probabilities(self):
        self.assertEqual(dealer_outcomes(17, True), (1, 0, 0, 0, 0, 0))
        self.assertEqual(dealer_outcomes(22, False), (0, 0, 0, 0, 0, 1))
        for upcard in range(1, 11):
            distribution = dealer_distribution(upcard)
            self.assertAlmostEqual(sum(distribution), 1.0)
            self.assertTrue(all(0 <= p <= 1 for p in distribution))

    def test_action_values_and_strategy(self):
        for state in states():
            self.assertTrue(all(-1.000000001 <= value <= 1.000000001 for value in action_values(state).values()))
        for total, dealer, soft, action in [(16, 10, False, "HIT"), (12, 4, False, "STAND"),
                                          (11, 10, False, "HIT"), (18, 9, True, "HIT"),
                                          (18, 2, True, "STAND"), (19, 6, True, "STAND")]:
            self.assertEqual(optimal_action(State(total, dealer, soft)), action)
        for dealer in range(1, 11):
            self.assertAlmostEqual(action_values(State(21, dealer, False))["STAND"],
                                   1 - dealer_distribution(dealer)[4])

    def test_values_against_independent_simulation(self):
        import gymnasium as gym
        env = gym.make("Blackjack-v1", natural=False, sab=False)
        samples = 12_000
        try:
            for state in (State(16, 10, False), State(18, 9, True)):
                for action in ("HIT", "STAND"):
                    total_return = 0.0
                    for i in range(samples):
                        env.reset(seed=8675309 if i == 0 else None)
                        env.unwrapped.player = [1, state.total - 11] if state.usable_ace else [10, state.total - 10]
                        env.unwrapped.dealer[0] = state.dealer
                        observation, result, done, _, _ = env.step(int(action == "HIT"))
                        while not done:
                            following = State(int(observation[0]), int(observation[1]), bool(observation[2]))
                            observation, result, done, _, _ = env.step(int(optimal_action(following) == "HIT"))
                        total_return += result
                    self.assertAlmostEqual(total_return / samples, action_values(state)[action], delta=0.04)
        finally:
            env.close()

    def test_parsing_and_rewards(self):
        state = State(16, 10, False)
        good = '{"reasoning":"Hitting offers a better expected return.","action":"HIT"}'
        self.assertAlmostEqual(reward(good, state), action_values(state)["HIT"])
        for bad in ('', 'HIT', '[]', '{}', 'null', '{"reasoning":" ","action":"HIT"}',
                    '{"reasoning":1,"action":"HIT"}', '{"reasoning":"x","action":"hit"}',
                    '{"reasoning":"x","action":"HIT","extra":1}',
                    '{"reasoning":"x","action":"HIT","action":"STAND"}',
                    '{"reasoning":"x","action":["HIT"]}'):
            self.assertIsNone(parse_completion(bad))
            self.assertEqual(reward(bad, state), -2)
        scores = blackjack_reward([[{"role": "assistant", "content": good}], "invalid"],
                                  total=[16, 16], dealer=[10, 10], usable_ace=[False, False])
        self.assertEqual(scores, [action_values(state)["HIT"], -2])
        self.assertNotEqual(prompt(state), prompt(state, "heldout"))
        self.assertNotIn(str(action_values(state)["HIT"]), str(prompt(state)))


if __name__ == "__main__":
    unittest.main()
