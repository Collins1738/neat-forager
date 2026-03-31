"""
inspect.py — Inspect and visualize the winner genome from winner.pkl
Prints stats and draws the neural network structure as a PNG.
"""

import pickle
import os
import sys
import neat
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'forager_config.txt')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'winner_network.png')

INPUT_LABELS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW',
                'vN', 'vNE', 'vE', 'vSE', 'vS', 'vSW', 'vW', 'vNW',
                'wN', 'wE', 'wS', 'wW']
OUTPUT_LABELS = ['↑ N', '→ E', '↓ S', '← W']


def draw_network(genome, config, output_path):
    """Draw the neural network as a clean diagram and save to PNG."""

    # Collect nodes
    input_keys = config.genome_config.input_keys   # e.g. [-1, -2, ..., -8]
    output_keys = config.genome_config.output_keys  # e.g. [0, 1, 2, 3]
    hidden_keys = [k for k in genome.nodes.keys() if k not in output_keys]

    n_inputs = len(input_keys)
    n_outputs = len(output_keys)
    n_hidden = len(hidden_keys)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor('#0f0f1a')
    fig.patch.set_facecolor('#0f0f1a')
    ax.axis('off')

    # Layer x positions
    x_in = 0.1
    x_hid = 0.5
    x_out = 0.9

    def y_positions(n, margin=0.1):
        if n == 1:
            return [0.5]
        return [margin + i * (1 - 2 * margin) / (n - 1) for i in range(n)]

    # Node positions
    pos = {}
    for i, k in enumerate(input_keys):
        pos[k] = (x_in, y_positions(n_inputs)[i])
    for i, k in enumerate(output_keys):
        pos[k] = (x_out, y_positions(n_outputs)[i])
    for i, k in enumerate(hidden_keys):
        pos[k] = (x_hid, y_positions(max(n_hidden, 1))[i] if n_hidden > 0 else 0.5)

    # Draw connections
    for cg in genome.connections.values():
        if not cg.enabled:
            continue
        if cg.key[0] not in pos or cg.key[1] not in pos:
            continue
        x0, y0 = pos[cg.key[0]]
        x1, y1 = pos[cg.key[1]]
        w = cg.weight
        color = '#50dc64' if w > 0 else '#dc5050'
        alpha = min(1.0, 0.2 + abs(w) / 10)
        lw = 0.5 + abs(w) / 6
        ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, linewidth=lw, zorder=1)

    # Draw nodes
    def draw_node(key, x, y, label, color, fontsize=9):
        circle = plt.Circle((x, y), 0.035, color=color, zorder=3, linewidth=1.5,
                             edgecolor='white')
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
                color='white', fontweight='bold', zorder=4)

    for i, k in enumerate(input_keys):
        x, y = pos[k]
        draw_node(k, x, y, INPUT_LABELS[i], '#3060c8')
        ax.text(x - 0.055, y, INPUT_LABELS[i], ha='right', va='center',
                fontsize=8, color='#aaaacc')

    for i, k in enumerate(output_keys):
        x, y = pos[k]
        draw_node(k, x, y, str(i), '#50a060')
        ax.text(x + 0.055, y, OUTPUT_LABELS[i], ha='left', va='center',
                fontsize=8, color='#aaaacc')

    for k in hidden_keys:
        x, y = pos[k]
        draw_node(k, x, y, str(k), '#8060c0')

    # Layer labels
    ax.text(x_in, 1.04, 'INPUTS\n(sensors)', ha='center', va='bottom',
            fontsize=9, color='#6688cc')
    if n_hidden > 0:
        ax.text(x_hid, 1.04, f'HIDDEN\n({n_hidden} nodes)', ha='center', va='bottom',
                fontsize=9, color='#9966cc')
    ax.text(x_out, 1.04, 'OUTPUTS\n(move)', ha='center', va='bottom',
            fontsize=9, color='#66aa77')

    # Legend
    green_patch = mpatches.Patch(color='#50dc64', label='Positive weight')
    red_patch = mpatches.Patch(color='#dc5050', label='Negative weight')
    ax.legend(handles=[green_patch, red_patch], loc='lower center',
              facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white',
              fontsize=8, ncol=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.15)
    ax.set_title('Winner Neural Network', color='white', fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"🖼️  Network diagram saved to: {output_path}")


def inspect():
    if not os.path.exists('winner.pkl'):
        print("❌ No winner.pkl found. Run train.py first.")
        sys.exit(1)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    with open('winner.pkl', 'rb') as f:
        winner = pickle.load(f)

    input_keys = config.genome_config.input_keys
    output_keys = config.genome_config.output_keys
    hidden_keys = [k for k in winner.nodes.keys() if k not in output_keys]
    enabled_conns = [c for c in winner.connections.values() if c.enabled]
    disabled_conns = [c for c in winner.connections.values() if not c.enabled]

    weights = [c.weight for c in enabled_conns]

    print("\n" + "="*50)
    print("  🏆 WINNER GENOME STATS")
    print("="*50)
    print(f"  Fitness:            {winner.fitness:.2f}")
    print(f"  Input nodes:        {len(input_keys)}")
    print(f"  Hidden nodes:       {len(hidden_keys)}")
    print(f"  Output nodes:       {len(output_keys)}")
    print(f"  Enabled connections:{len(enabled_conns)}")
    print(f"  Disabled connections:{len(disabled_conns)}")
    if weights:
        print(f"  Avg weight:         {np.mean(weights):.3f}")
        print(f"  Max weight:         {max(weights):.3f}")
        print(f"  Min weight:         {min(weights):.3f}")
    print("="*50)

    if hidden_keys:
        print(f"\n  Hidden node keys: {hidden_keys}")
    else:
        print("\n  No hidden nodes — direct input→output network")

    print("\n  Connections (enabled):")
    for c in sorted(enabled_conns, key=lambda x: x.key):
        src = INPUT_LABELS[-c.key[0] - 1] if c.key[0] in input_keys else str(c.key[0])
        dst = OUTPUT_LABELS[c.key[1]] if c.key[1] in output_keys else str(c.key[1])
        bar = '█' * int(abs(c.weight) * 2)
        sign = '+' if c.weight > 0 else '-'
        print(f"    {src:6} → {dst:6}  {sign}{abs(c.weight):.3f}  {bar}")

    draw_network(winner, config, OUTPUT_PATH)


if __name__ == '__main__':
    inspect()
