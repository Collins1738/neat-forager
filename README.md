# 🐛 NEAT Foragers

Agents learn to find food and survive across generations using the NEAT algorithm.

## Setup

```bash
pip install neat-python pygame
```

## Run

### Train fast (no visuals, prints stats each gen)
```bash
python train.py
```

### Train + watch best agent each generation
```bash
python visualize.py --train
```

### Watch the saved winner
```bash
python visualize.py
```

## Controls (during visualization)
- `SPACE` — pause/unpause
- `ESC` — quit

## How it works

- Each agent has 8 directional sensors pointing N/NE/E/SE/S/SW/W/NW
- Sensors return how close food is in that direction (0 = nothing, 1 = right there)
- The brain (neural net) takes 8 inputs → outputs 4 values (N/E/S/W movement)
- Agents that eat more food and survive longer = higher fitness
- Top performers breed into the next generation
- Neural net structure grows over time (NEAT adds nodes/connections through evolution)

## Files
- `simulation.py` — core agent logic (sensors, movement, scoring)
- `train.py` — headless training, saves winner to `winner.pkl`
- `visualize.py` — pygame visualization
- `forager_config.txt` — NEAT hyperparameters (tune this!)
