<!-- LOGO -->
<br />
<h1>
<p align="center">
  <img src="./assets/balatro.png" alt="Logo" width="140" height="140">
  <br>Balatro RL Agent
</h1>
  <p align="center">
    A live Balatro card-play agent with imitation learning, exact scoring, and an RL-ready Gymnasium bridge.
    <br />
    </p>
</p>
<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#credits">Credits</a>
</p>  

## Overview

This project builds and evaluates AI agents for Balatro's in-blind card decisions by:

- Injecting Lua code into the Love2D game engine using the Lovely Injector mod system
- Monitoring game state in real-time through direct memory access (money, chips, hands remaining, discards remaining, current hand cards)
- Converting visible game state into versioned feature vectors and planner states
- Training behavioral-cloning and fixed-vocabulary supervised candidate policies
- Reranking supported play actions with a deterministic Balatro scoring engine
- Exposing a Gymnasium environment for later PPO fine-tuning
- Executing actions through a bidirectional file-based communication protocol (JSON commands for playing or discarding cards)

The live planner observes only normally visible information. It ranks legal play
and discard actions, then optionally reranks play candidates with the scoring
engine. The current checkpoint is a supervised imitation baseline, not a
finished reinforcement-learning policy; PPO remains a later fine-tuning stage.

### Scope: blind-round card score only

The RL model is **only** for maximizing card score during the actual blind rounds in each ante:

- **In scope:** Small blind, big blind (and boss blind) — the rounds where you play or discard cards to meet the chip target. The agent learns which cards to play or discard each turn to maximize chips.
- **Out of scope:** All power-up and meta decisions: shop purchases, joker/tarot/planet selection, booster packs, and blind skipping. Automatic progression can leave a shop without buying and select the required next blind, but it does not make deck-building decisions.

So the goal is to master **in-round card selection** (play vs discard, which cards to pick), not deck building or power-up choices.

## How It Works
- Initial Setup:

I am using the steam version of the game so follow the Lovely Injector instructions first. After that set the launch options in steam as --dump-all.

Essentially what this does is it uses the Lovely Injector to dump all the game files and then we can search them to see how game info is handled. To find your dump folder on **Windows** go to _AppData\Roaming\Balatro\Mods\lovely\dump_. Since the dump contains the game files we can just open this into VS Code and do _Ctrl+Shift+F_ or _Cmd+Shift+F_ to search through all the game files at once to see stuff like: 1. How score is handled, 2. What happens when you play a hand, etc.

Once you have openned the **dump** folder in VS Code, you will see a bunch of Lua files. These are all the important game files that show how logic is handled.

We will make a Symlink between the bridge folder here and the bridge folder in the Lovely Mods directory.

## The Bridge: Communication Layer

The bridge is a **Lua-based communication layer** that enables bidirectional communication between Python agents and the Balatro game. It acts as an intermediary, allowing an agent to observe game state and execute actions within the game.

### What the Bridge Does

The bridge (`bridge/main.lua`) is injected into the game's runtime using the Lovely Injector mod system. It operates by hooking into the game's main update loop and performs the following functions:

#### 1. **Game State Monitoring**
- Hooks into `Game:update(dt)` to run every frame
- Monitors key game state variables:
  - Current game phase/state (`G.STATE`)
  - Money (`G.GAME.dollars`)
  - Chips (current and blind target)
  - Hands remaining (`G.GAME.current_round.hands_left`)
  - Discards remaining (`G.GAME.current_round.discards_left`)
  - Current hand cards and their properties

#### 2. **Event-Based State Snapshotting**
The bridge uses an event-driven approach to capture game state only when relevant changes occur. It triggers a state dump when:
- **State Transition**: The game enters the `SELECTING_HAND` state (state 1), indicating the player can make decisions
- **Hand Played**: The number of hands remaining decreases
- **Cards Discarded**: The number of discards remaining decreases
- **Hand Size Changed**: New cards are drawn or cards are removed

When triggered, the bridge outputs a formatted snapshot containing:
- Current phase/state
- Money and chip values
- Hands and discards remaining
- Complete hand information (card indices, values, and suits)

#### 3. **Command Execution**
The bridge continuously polls for commands from the Python agent via a JSON file (`command.json`). When a command file is detected:

1. **Reads** the command file containing:
   - `action`: Either `"play"` or `"discard"`
   - `cards`: Array of card indices (1-based) to select

2. **Selects Cards**: Highlights the specified cards in the game's hand
   - Uses `G.hand:unhighlight_all()` to clear previous selections
   - Uses `G.hand:add_to_highlighted(card)` for each specified card index

3. **Executes Action**: Calls the appropriate game function:
   - `G.FUNCS.play_cards_from_highlighted()` for playing cards
   - `G.FUNCS.discard_cards_from_highlighted()` for discarding cards

4. **Cleans Up**: Deletes the command file after processing to prevent re-execution

#### 4. **State Tracking**
The bridge maintains internal state tracking to detect changes:
- `last_state`: Previous game state value
- `last_hands`: Previous hands remaining count
- `last_discards`: Previous discards remaining count
- `last_card_count`: Previous hand size

This prevents redundant state dumps and ensures the agent only receives updates when meaningful changes occur.

### Bridge Configuration

The bridge is loaded into the game via `bridge/lovely.toml`, which configures the Lovely Injector to:
- Patch the game's `main.lua` file
- Inject `require('bridge/main')` after the `require "challenges"` line
- Ensure the bridge code runs alongside the game's main logic

### Communication Protocol

The bridge implements a **file-based communication protocol**:

**Agent → Game (Actions)**:
```
Python agent writes → command.json → Bridge reads → Bridge executes → Game state changes
```

**Game → Agent (Observations)**:
```
Game state changes → Bridge detects → Bridge writes state.json → Agent reads state.json
```
The bridge writes `state.json` (and still prints to console) whenever state changes in `SELECTING_HAND`. The Python agent reads this file to get the current observation.

**State file format** (`state.json`):
```json
{
  "seq": 42,
  "state_version": 2,
  "phase": 1,
  "money": 150,
  "chips": 0,
  "blind_chips": 300,
  "hands_left": 4,
  "discards_left": 3,
  "round_result": "",
  "run": {
    "seed": "ABC123",
    "ante": 1,
    "deck_remaining": 44,
    "deck_total": 52
  },
  "blind": {
    "key": "bl_small",
    "name": "Small Blind",
    "type": "Small",
    "boss": false,
    "disabled": false,
    "debuff": {}
  },
  "hand": [
    {"index": 1, "value": "A", "suit": "Spades"},
    {"index": 2, "value": "K", "suit": "Hearts"}
  ]
}
```

The command file format:
```json
{
  "action": "play" | "discard",
  "cards": [1, 2, 3]  // 1-based indices
}
```

With automatic progression enabled, Python may also send:

```json
{"action": "advance", "cards": []}
```

The bridge waits for Balatro's real UI controls, presses Cash Out after a win,
leaves the shop without purchases, and selects the actual Small, Big, or Boss
Blind button. After a loss it starts a new run with the default Red Deck.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Python Policy / Planner                   │
│  ┌──────────────────┐              ┌──────────────────┐         │
│  │  Policy Network  │              │  Value Network   │         │
│  │  (Actor)         │              │  (Critic)        │         │
│  └──────────────────┘              └──────────────────┘         │
│           │                                  │                  │
│           └──────────┬───────────────────────┘                  │
│                      │                                          │
│              ┌───────▼─────────┐                                │
│              │ Action Selector │                                │
│              └───────┬─────────┘                                │
└──────────────────────┼──────────────────────────────────────────┘
                       │
                       │ writes command.json
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                    File System                                  │
│              state.json (bridge→agent)  command.json (agent→bridge) │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       │ reads & deletes
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                    Bridge (Lua)                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Game.update() Hook                                      │   │
│  │  ┌──────────────────┐  ┌─────────────────────────────┐   │   │
│  │  │ check_for_       │  │  State Change Detection     │   │   │
│  │  │ commands()       │  │  - State transitions        │   │   │
│  │  │                  │  │  - Hand/discard changes     │   │   │
│  │  │  • Read JSON     │  │  - Card count changes       │   │   │
│  │  │  • Parse action  │  │                             │   │   │
│  │  │  • Select cards  │  │  ┌───────────────────────┐  │   │   │
│  │  │  • Execute cmd   │  │  │ dump_game_state()     │  │   │   │
│  │  │  • Delete file   │  │  │ write_state_json()    │  │   │   │
│  │  └──────────────────┘  │  │  • Print + state.json │  │   │   │
│  │                        │  └───────────────────────┘  │   │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       │ hooks into & calls game functions
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                    Balatro Game (Love2D)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Game State (G.*)                                        │   │
│  │  • G.STATE (current phase)                               │   │
│  │  • G.GAME.dollars (money)                                │   │
│  │  • G.GAME.chips (score)                                  │   │
│  │  • G.hand.cards[] (current hand)                         │   │
│  │  • G.GAME.current_round (hands/discards left)            │   │
│  │                                                          │   │
│  │  Game Functions                                          │   │
│  │  • G.FUNCS.play_cards_from_highlighted()                 │   │
│  │  • G.FUNCS.discard_cards_from_highlighted()              │   │
│  │  • G.hand:add_to_highlighted(card)                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

Observation Flow (Game → Agent):
Game State → Bridge detects change → Bridge writes state.json → Agent reads state.json

Action Flow (Agent → Game):
Agent decides action → Agent writes command.json → Bridge reads → Bridge selects cards → Bridge executes → Game updates
```

### Key Design Decisions

1. **File-Based Communication**: Uses temporary JSON files instead of sockets/process communication for simplicity and compatibility with the game's Lua runtime
2. **Event-Driven Snapshotting**: Only captures state when meaningful changes occur, reducing noise and improving efficiency
3. **State Tracking**: Maintains previous values to detect transitions rather than polling continuously
4. **Safe Execution**: Uses `pcall()` to wrap state dumps, preventing crashes from unexpected game state
5. **Command Cleanup**: Deletes command files immediately after reading to prevent duplicate executions

## Running the Python environment

1. **Symlink** the repo `bridge` folder into the Lovely Mods directory and run Balatro (with the bridge injected) so that `state.json` is written to `bridge/`.
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run the example** (random agent, 20 steps): `python run_env_example.py`  
   Optionally set `BALATRO_BRIDGE_DIR` to the full path of the bridge folder if it differs from the repo.
4. Use **Gymnasium** in your own scripts:
   ```python
   from env import BalatroEnv
   env = BalatroEnv(bridge_dir="path/to/bridge", observation_version=2)
   obs, info = env.reset()
   action = env.action_space.sample()  # or your policy
   obs, reward, term, trunc, info = env.step(action)
   ```
   **Observation v1** is the original 60-value vector and remains the default for compatibility with existing models. **Observation v2** is a 915-value visible-information vector with blind/debuff context, chip progress, run resources, poker-pattern summaries, enriched cards, hand levels, jokers, and consumables. The run seed is deliberately excluded from both policy inputs. **Reward** is shaped: chip delta + discard penalty + one hand-type bonus per scored play, with terminal bonuses for winning or losing a blind. **Action** is `MultiDiscrete`: play (0) vs discard (1), then 8 bits for which cards (1-based indices). When no discards remain, Python converts discard requests to play and Lua rejects invalid commands as a final safeguard.

## Recording expert data (for Imitation Learning)

To collect (observation, action) pairs while you play, run the recorder and type actions at the prompt:

```bash
python record_expert.py --output expert_data.jsonl
```

With Balatro on a hand-selection screen, the script shows the hand and waits for input. Type e.g. `play 1,2,3` or `discard 4,5` (1-based indices; max 5 cards for play). After the bridge confirms the action, the pair is appended as one JSON line: `{"obs": [...], "action": [...]}` (same format as the env).

New files default to observation v2. If the output file already contains v1
records, the recorder detects that and continues in v1 so dimensions are never
mixed. Use a new filename to begin a v2 dataset:

```bash
python record_expert.py --output expert_data_v2.jsonl
```

Train a behavioral-cloning policy after recording at least a small demonstration set:

```bash
python train_bc.py --data expert_data.jsonl --model-out models/balatro_bc.zip
```

The trainer validates every record, keeps a validation split, prints action/card
accuracy, and saves the best policy plus `balatro_bc.metrics.json`. The saved
model is a normal Stable Baselines3 PPO model, so it can later continue with PPO
training.

Run the cloned policy in a fresh blind:

```bash
python play_bc.py --model models/balatro_bc.zip
```

To run multiple card-play episodes without buying anything from shops, enable
automatic progression:

```bash
python play_bc.py --model models/balatro_bc.zip --auto-advance --episodes 3
```

Automatic progression follows the real game flow: it presses Cash Out after a
win, leaves shops without purchases, and selects the actual next Small, Big, or
Boss Blind button. After a loss it follows the Game Over `New Run` flow,
selects the default Red Deck, presses Play, and selects the Small Blind. It is
opt-in; normal manual play is unchanged.

Expert records retain the structured raw state (seed/run context, active blind
and debuff, enriched cards, jokers, consumables, and deck counts). Playback
prints the seed and appends each seed, blind outcome, and move sequence to
`play_history.jsonl` for reproducibility. The seed is metadata only and is never
passed to the policy. Training and playback infer v1 or v2 automatically from
the dataset/model dimension; mixed-version datasets are rejected.

For a first pipeline check, about 100 recorded decisions is enough. A useful
policy will need a larger and more varied dataset covering plays, discards,
different hand shapes, and decisions near the end of a blind.

## Importing BalatroBench

Normalize a downloaded BalatroBench `runs/` tree without copying screenshots,
the repeated game guide, or hidden deck order:

```bash
python import_balatrobench.py --source "E:\balatro-data\runs\runs" --output data/balatrobench
```

The importer writes four linked JSONL tables plus `manifest.json`:

- `runs.jsonl`: model, strategy, seed, deck, stake, and final outcome metadata.
- `states.jsonl`: structured post-action snapshots with seed removed, face-down
  hand identities masked, and remaining-deck cards sorted canonically.
- `request_states.jsonl`: every rendered pre-action game-state block, excluding
  the seed and repeated tool documentation.
- `transitions.jsonl`: tool calls linked to request states and, where alignment
  is exact, structured pre/post states.

`bc_candidate` marks play/discard records with exact structured transitions.
`text_bc_candidate` marks well-formed play/discard calls with a rendered
`SELECTING_HAND` pre-state. Runs containing failed calls remain fully indexed,
but the importer does not guess shifted structured-state alignment.

Build compact in-blind planner examples from those normalized tables:

```bash
python build_planner_dataset.py --input data/balatrobench
```

This writes `planner_examples.jsonl` and `planner_manifest.json`. Each example
contains a parsed visible pre-move state, the play/discard action and selected
card snapshots, run outcome metadata, and data-quality flags. The seed is kept
only under provenance for replay and is not part of the policy state. Where an
exact structured pre-state exists, the manifest reports field-by-field parser
match rates instead of silently trusting the rendered-text parser.

Run the repeatable integrity audit before using a regenerated file for training:

```bash
python audit_planner_dataset.py --data data/balatrobench/planner_examples.jsonl
```

The audit checks legal actions, selected-card joins, all 12 poker-hand records,
hidden card/Joker masking, modifier vocabularies, and seed exclusion from the
policy state. A passing audit means the examples are structurally sound; it does
not mean the source LLM actions are optimal. Identical visible states must stay
in one split, and final agent evaluation must use fresh live-game seeds rather
than randomly shuffled decisions from these five recorded seeds.

Prepare the audited decisions for supervised planner training:

```bash
python prepare_planner_training_data.py
```

Preparation rejects unusable rows, merges identical visible states, aggregates
conflicting actions into probability targets, and gives each source model equal
vote weight. The deterministic train/validation split uses only a hash of the
visible state; it does not use the seed. These remain unrated LLM targets until
the legal-action scorer and rollout evaluator can assign action values.

Train the frozen pre-rollout comparison baseline:

```bash
python train_planner_baseline.py --model-out models/planner_llm_baseline.pt --epochs 30
```

The policy encodes visible planner state and ranks a fixed vocabulary of 3,170
play/discard subsets, with illegal actions masked before training and inference.
It saves the best validation-loss checkpoint plus a sibling `.metrics.json`
file. The seed-42 baseline selects epoch 6 and reaches `4.7244` validation NLL,
`12.92%` exact consensus accuracy, `66.81%` play/discard-kind accuracy, and
`29.17%` top-five target coverage on 720 validation states. Future rollout-value
training should keep this split, encoder, model, and evaluation protocol fixed.

Run that planner against the live game:

```bash
python play_planner.py --model models/planner_llm_baseline.pt --auto-advance --episodes 3
```

Restart Balatro after pulling or editing `bridge/main.lua`, open the game, and
leave it at any normal run/blind screen. `--auto-advance` selects the default
Red Deck after a loss, presses Cash Out after wins, leaves shops without
purchases, selects the next blind (including Boss Blinds), and continues between
the requested episodes. The runner logs the seed, blind, policy probabilities,
moves, and result to `planner_play_history.jsonl`. Omit `--auto-advance` to play
one already-open blind. By default, the model decides play versus discard and
the deterministic scoring engine reranks supported plays to the highest-scoring
visible subset. Pass `--policy-only` to reproduce the raw imitation-policy baseline. This
checkpoint was trained to imitate unrated LLM actions, so discard behavior is a
baseline until future-draw rollouts provide value targets.

## Base Scoring Engine

`ai_agent/scoring_engine.py` enumerates every legal 1-5 card play/discard set,
classifies all 12 Balatro hand types, identifies scored cards, and applies base
hand levels plus deterministic playing-card effects. Face-down selected cards
are never guessed. Unimplemented Jokers and boss rules are returned as explicit
`unsupported_effects`, so a baseline estimate cannot be mistaken for an exact
label.

Validate exact-eligible predictions against observed BalatroBench scores:

```bash
python validate_scoring_engine.py
```

The current base validation covers supported plays without active Joker or boss
modifiers. The first phase-aware Joker batch adds common conditional Chips/Mult,
per-scored-card effects, retriggers, held-card effects, Smeared Joker, Splash,
dynamic current values, and Joker editions. Unsupported copying, probability,
classification-changing, or stateful effects remain explicit rather than being
approximated silently. Stochastic discard rollouts are a later engine layer.

## Tests

Run the complete offline test suite without starting Balatro:

```bash
python -m unittest discover -s tests -v
```

Changes to `bridge/main.lua` still require a Balatro restart and a live smoke
test because Lovely loads the Lua bridge at game startup.

## Roadmap
- [x] Bidirectional Communication Bridge (Lua/Python)
- [x] State Reflection (Direct Memory Access)
- [x] Feature Encoding (Numerical Vectorization)
- [x] Gymnasium Environment Wrapper
- [x] Behavioral Cloning Training And Inference Pipeline
- [x] Import And Validate BalatroBench In-Blind Decisions
- [x] Implement Legal-Action Enumeration And Conservative Deterministic Scoring
- [x] Train Pre-Rollout Candidate Policy Baseline
- [ ] Generate Rollout Values And Retrain Candidate Policy
- [ ] PPO Fine-Tuning

## Credits
- Credit to [@ethangreen-dev](https://github.com/ethangreen-dev/lovely-injector) for the Love2D Injector code.
- Credit to [Stable-Baseline3](https://github.com/DLR-RM/stable-baselines3) for PPO implementation
- Credit to [Gymnasium](https://gymnasium.farama.org/) for the RL environment framework

## License

This project is open source and available under the MIT License.
