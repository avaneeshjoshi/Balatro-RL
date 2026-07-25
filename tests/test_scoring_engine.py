import unittest

from ai_agent.scoring_engine import (
    HiddenCardError,
    classify_hand,
    enumerate_legal_actions,
    score_play,
)


BASE_HANDS = {
    "Flush Five": {"chips": 160, "mult": 16},
    "Flush House": {"chips": 140, "mult": 14},
    "Five of a Kind": {"chips": 120, "mult": 12},
    "Straight Flush": {"chips": 100, "mult": 8},
    "Four of a Kind": {"chips": 60, "mult": 7},
    "Full House": {"chips": 40, "mult": 4},
    "Flush": {"chips": 35, "mult": 4},
    "Straight": {"chips": 30, "mult": 4},
    "Three of a Kind": {"chips": 30, "mult": 3},
    "Two Pair": {"chips": 20, "mult": 2},
    "Pair": {"chips": 10, "mult": 2},
    "High Card": {"chips": 5, "mult": 1},
}


def card(rank=None, suit=None, **values):
    result = {
        "hidden": False,
        "stone": False,
        "rank": rank,
        "suit": suit,
        "enhancements": [],
        "editions": [],
        "seals": [],
        "debuffed": False,
        "extra_chips": 0,
    }
    result.update(values)
    return result


def state(cards, hands=4, discards=3):
    return {
        "resources": {
            "hands": {"remaining": hands},
            "discards": {"remaining": discards},
        },
        "hand": {"count": len(cards), "cards": cards},
        "poker_hands": BASE_HANDS,
        "jokers": {"count": 0, "cards": []},
        "blind": {"effect": ""},
    }


class ScoringEngineTests(unittest.TestCase):
    def test_enumerates_all_play_and_discard_sets(self):
        game = state([card("A", "S") for _ in range(8)])
        actions = enumerate_legal_actions(game)

        self.assertEqual(len(actions), 436)
        self.assertEqual(sum(action.kind == "play" for action in actions), 218)
        self.assertEqual(sum(action.kind == "discard" for action in actions), 218)
        self.assertTrue(all(1 <= len(action.card_indices) <= 5 for action in actions))

    def test_omits_discards_when_none_remain(self):
        actions = enumerate_legal_actions(state([card("A", "S") for _ in range(8)], discards=0))
        self.assertEqual(len(actions), 218)
        self.assertTrue(all(action.kind == "play" for action in actions))

    def test_classifies_all_secret_and_standard_hands(self):
        cases = [
            ("Flush Five", [card("A", "S") for _ in range(5)]),
            ("Flush House", [card("7", "H") for _ in range(3)] + [card("4", "H") for _ in range(2)]),
            ("Five of a Kind", [card("A", suit) for suit in ["S", "H", "D", "C", "S"]]),
            ("Straight Flush", [card(rank, "D") for rank in ["A", "K", "Q", "J", "T"]]),
            ("Four of a Kind", [card("5", suit) for suit in ["S", "H", "D", "C"]] + [card("2", "D")]),
            ("Full House", [card("6", suit) for suit in ["S", "H", "D"]] + [card("K", "C"), card("K", "H")]),
            ("Flush", [card(rank, "D") for rank in ["J", "9", "8", "4", "3"]]),
            ("Straight", [card(rank, suit) for rank, suit in zip(["T", "9", "8", "7", "6"], ["D", "S", "H", "D", "C"])]),
            ("Three of a Kind", [card("Q", suit) for suit in ["C", "S", "H"]] + [card("9", "H"), card("2", "S")]),
            ("Two Pair", [card("J", "H"), card("J", "S"), card("3", "C"), card("3", "S"), card("2", "H")]),
            ("Pair", [card("T", "S"), card("T", "H"), card("8", "S"), card("7", "H"), card("4", "C")]),
            ("High Card", [card("K", "D"), card("Q", "D"), card("7", "S"), card("4", "S"), card("3", "H")]),
        ]
        for expected, cards in cases:
            with self.subTest(expected):
                self.assertEqual(classify_hand(state(cards), range(5)).hand_type, expected)

    def test_royal_flush_scores_as_straight_flush(self):
        game = state([card(rank, "D") for rank in ["A", "K", "Q", "J", "T"]])
        result = score_play(game, range(5))
        self.assertEqual(result.hand_type, "Straight Flush")
        self.assertEqual(result.score, 1208)
        self.assertTrue(result.exact)

    def test_pair_scores_only_pair_and_always_scores_stone(self):
        cards = [
            card("T", "S"),
            card("T", "H"),
            card("8", "S"),
            card(None, None, stone=True),
            card("4", "C"),
        ]
        result = score_play(state(cards), range(5))
        self.assertEqual(result.hand_type, "Pair")
        self.assertEqual(result.scoring_indices, (0, 1, 3))
        self.assertEqual(result.score, 160)

    def test_wild_card_completes_flush(self):
        cards = [card(rank, "H") for rank in ["A", "Q", "9", "5"]]
        cards.append(card("2", "C", enhancements=["WILD"]))
        self.assertEqual(classify_hand(state(cards), range(5)).hand_type, "Flush")

    def test_card_effects_and_held_steel_are_applied(self):
        cards = [
            card("A", "S", enhancements=["BONUS"], editions=["FOIL"], seals=["RED"]),
            card("K", "H", enhancements=["STEEL"]),
        ]
        result = score_play(state(cards), [0])
        # High Card starts 5x1. The retriggered Ace adds (11+30+50)*2,
        # then the held Steel card multiplies Mult by 1.5.
        self.assertEqual(result.chips, 187)
        self.assertEqual(result.mult, 1.5)
        self.assertEqual(result.score, 280)
        self.assertTrue(result.exact)

    def test_permanent_extra_chips_are_scored(self):
        result = score_play(state([card("A", "S", extra_chips=20)]), [0])
        self.assertEqual(result.chips, 36)
        self.assertEqual(result.score, 36)

    def test_green_joker_updates_before_scoring_current_hand(self):
        game = state([card("A", "D")])
        game["jokers"] = {
            "count": 1,
            "cards": [
                {
                    "name": "Green Joker",
                    "effect": "+1 Mult per hand played -1 Mult per discard (Currently +5 Mult)",
                }
            ],
        }
        self.assertEqual(score_play(game, [0]).score, 112)

    def test_raised_fist_resolves_after_held_steel(self):
        game = state(
            [
                card("T", "H"),
                card("A", "S", enhancements=["STEEL"]),
                card("J", "D"),
            ]
        )
        game["jokers"] = {
            "count": 1,
            "cards": [{"name": "Raised Fist", "effect": "Double lowest held rank"}],
        }
        result = score_play(game, [0])
        # (base 1 Mult * held Steel 1.5) + double Jack rank (20) = 21.5.
        self.assertEqual(result.mult, 21.5)
        self.assertEqual(result.score, 322)

    def test_unsupported_jokers_make_base_result_explicitly_inexact(self):
        game = state([card("A", "S")])
        game["jokers"] = {
            "count": 1,
            "cards": [{"name": "Blueprint", "effect": "Copies ability of Joker to the right"}],
        }
        result = score_play(game, [0])
        self.assertFalse(result.exact)
        self.assertEqual(result.unsupported_effects, ("joker:Blueprint",))

    def test_independent_and_card_trigger_jokers_score_in_order(self):
        game = state([card("8", "S")])
        game["jokers"] = {
            "count": 2,
            "cards": [
                {"name": "Even Steven", "effect": "+4 Mult for even ranks"},
                {"name": "Joker", "effect": "+4 Mult"},
            ],
        }
        result = score_play(game, [0])
        self.assertEqual(result.chips, 13)
        self.assertEqual(result.mult, 9)
        self.assertEqual(result.score, 117)
        self.assertTrue(result.exact)

    def test_splash_hanging_chad_and_raised_fist_phases(self):
        game = state([card("A", "S"), card("K", "H"), card("2", "D")])
        game["jokers"] = {
            "count": 3,
            "cards": [
                {"name": "Splash", "effect": "Every played card counts in scoring"},
                {"name": "Hanging Chad", "effect": "Retrigger first played scoring card twice"},
                {"name": "Raised Fist", "effect": "Double lowest held rank to Mult"},
            ],
        }
        result = score_play(game, [0, 1])
        self.assertEqual(result.scoring_indices, (0, 1))
        self.assertEqual(result.chips, 48)
        self.assertEqual(result.mult, 5)
        self.assertEqual(result.score, 240)
        self.assertTrue(result.exact)

    def test_smeared_joker_combines_suit_colours_for_flush(self):
        cards = [card(rank, suit) for rank, suit in zip(["A", "Q", "9", "5", "2"], ["H", "D", "H", "D", "H"])]
        game = state(cards)
        game["jokers"] = {
            "count": 1,
            "cards": [{"name": "Smeared Joker", "effect": "Hearts and Diamonds count as same suit"}],
        }
        self.assertEqual(score_play(game, range(5)).hand_type, "Flush")

    def test_face_down_selection_refuses_to_guess(self):
        game = state([card(None, None, hidden=True)])
        with self.assertRaises(HiddenCardError):
            classify_hand(game, [0])


if __name__ == "__main__":
    unittest.main()
