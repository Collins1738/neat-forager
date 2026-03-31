"""
visualize.py — Watch the best agent live using pygame.
Run after training: python visualize.py
Or train + watch: python visualize.py --train
"""

import pygame
import neat
import pickle
import os
import sys
from simulation import GRID_SIZE, INITIAL_ENERGY

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'forager_config.txt')

# Display settings
CELL_SIZE = 22
PANEL_WIDTH = 280
WIDTH = GRID_SIZE * CELL_SIZE + PANEL_WIDTH
HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 15

# Colors
BG = (18, 18, 28)
GRID_LINE = (30, 30, 45)
FOOD_COLOR = (80, 220, 100)
AGENT_COLOR = (80, 160, 255)
AGENT_TRAIL = (40, 80, 140)
PANEL_BG = (12, 12, 22)
TEXT_COLOR = (200, 200, 220)
DIM_TEXT = (100, 100, 130)
BAR_BG = (40, 40, 60)
BAR_ENERGY = (80, 200, 120)
BAR_LOW = (220, 80, 80)
SENSOR_COLOR = (255, 200, 50, 60)


def draw_panel(surface, font, small_font, generation, fitness, energy, food_eaten, step, steps_total, species_count, best_fitness, max_energy=None):
    panel_x = GRID_SIZE * CELL_SIZE
    pygame.draw.rect(surface, PANEL_BG, (panel_x, 0, PANEL_WIDTH, HEIGHT))
    pygame.draw.line(surface, GRID_LINE, (panel_x, 0), (panel_x, HEIGHT), 2)

    y = 20
    line_h = 26
    small_h = 22

    def label(text, val, color=TEXT_COLOR, small=False):
        nonlocal y
        f = small_font if small else font
        lbl = f.render(text, True, DIM_TEXT)
        val_surf = f.render(str(val), True, color)
        surface.blit(lbl, (panel_x + 16, y))
        surface.blit(val_surf, (panel_x + PANEL_WIDTH - val_surf.get_width() - 16, y))
        y += small_h if small else line_h

    title = font.render("🐛 NEAT Foragers", True, (120, 180, 255))
    surface.blit(title, (panel_x + 16, y))
    y += 36

    pygame.draw.line(surface, GRID_LINE, (panel_x + 10, y), (panel_x + PANEL_WIDTH - 10, y))
    y += 14

    label("Generation", generation)
    label("Step", f"{step}/{steps_total}")
    label("Food eaten", food_eaten, (80, 220, 100))
    fit_color = (80, 220, 100) if fitness >= 0 else (220, 80, 80)
    label("Fitness", f"{fitness:.1f}", fit_color)
    label("Best fitness", f"{best_fitness:.1f}", (255, 200, 80))
    label("Species", species_count, small=True)

    y += 10
    pygame.draw.line(surface, GRID_LINE, (panel_x + 10, y), (panel_x + PANEL_WIDTH - 10, y))
    y += 14

    # Energy bar
    energy_label = font.render("Energy", True, DIM_TEXT)
    surface.blit(energy_label, (panel_x + 16, y))
    y += 24
    bar_w = PANEL_WIDTH - 32
    bar_h = 14
    pygame.draw.rect(surface, BAR_BG, (panel_x + 16, y, bar_w, bar_h), border_radius=4)
    _max_e = max_energy or INITIAL_ENERGY
    fill = int(bar_w * max(0, energy) / _max_e)
    bar_color = BAR_ENERGY if energy > _max_e * 0.3 else BAR_LOW
    if fill > 0:
        pygame.draw.rect(surface, bar_color, (panel_x + 16, y, fill, bar_h), border_radius=4)
    y += bar_h + 20

    pygame.draw.line(surface, GRID_LINE, (panel_x + 10, y), (panel_x + PANEL_WIDTH - 10, y))
    y += 14

    hint = small_font.render("SPACE = pause  ESC = quit", True, DIM_TEXT)
    surface.blit(hint, (panel_x + 16, HEIGHT - 30))


def run_visual(net, generation=1, stats=None):
    from simulation import run_simulation, STEPS

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT Foragers")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 15)
    small_font = pygame.font.SysFont("monospace", 12)

    steps_total = 1000
    paused = [False]
    running = [True]
    food_eaten = [0]
    best_fitness = [0.0]
    prev_fitness = [0.0]

    def step_callback(state):
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running[0] = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running[0] = False
                elif event.key == pygame.K_SPACE:
                    paused[0] = not paused[0]

        if not running[0]:
            return False  # stop simulation

        while paused[0]:
            clock.tick(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running[0] = False
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running[0] = False
                        return False
                    elif event.key == pygame.K_SPACE:
                        paused[0] = False

        agent = state['agent']
        food = state['food']
        energy = state['energy']
        fitness = state['fitness']
        step = state['step']
        trail = state['trail']

        # Count food eaten by tracking fitness jumps of +10
        if fitness - prev_fitness[0] >= 9.9:
            food_eaten[0] += 1
        prev_fitness[0] = fitness

        # --- Draw ---
        screen.fill(BG)

        for i in range(GRID_SIZE + 1):
            pygame.draw.line(screen, GRID_LINE, (i * CELL_SIZE, 0), (i * CELL_SIZE, HEIGHT))
            pygame.draw.line(screen, GRID_LINE, (0, i * CELL_SIZE), (GRID_SIZE * CELL_SIZE, i * CELL_SIZE))

        for i, (tx, ty) in enumerate(trail):
            alpha = int(180 * (i / len(trail))) if trail else 0
            s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            s.fill((*AGENT_TRAIL, alpha))
            screen.blit(s, (tx * CELL_SIZE, ty * CELL_SIZE))

        for fx, fy in food:
            cx = fx * CELL_SIZE + CELL_SIZE // 2
            cy = fy * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(screen, FOOD_COLOR, (cx, cy), CELL_SIZE // 3)

        ax = agent[0] * CELL_SIZE + CELL_SIZE // 2
        ay = agent[1] * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, AGENT_COLOR, (ax, ay), CELL_SIZE // 2 - 2)
        pygame.draw.circle(screen, (150, 200, 255), (ax, ay), CELL_SIZE // 4)

        best_fitness[0] = max(best_fitness[0], fitness)
        draw_panel(screen, font, small_font, generation, round(fitness, 1),
                   energy, food_eaten[0], step + 1, steps_total, "—", best_fitness[0])

        pygame.display.flip()

    run_simulation(net, step_callback=step_callback, steps=1000)
    pygame.quit()


def watch_winner():
    winner = 'winner.pkl'
    if not os.path.exists(winner):
        print(f"❌ No {winner} found. Run: python train.py first")
        sys.exit(1)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    with open(winner, 'rb') as f:
        winner = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    print(f"🎬 Watching winner (fitness: {winner.fitness:.2f})")
    run_visual(net, generation="winner")


def train_and_watch():
    """Train with live visualization of best agent each generation."""
    import neat
    from multiprocessing import Pool
    from train import eval_genome

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    generation = [0]

    def eval_and_show(genomes, config):
        generation[0] += 1

        # Evaluate all genomes in parallel (averaged over 3 runs)
        with Pool() as pool:
            results = pool.map(eval_genome, [(gid, g, config) for gid, g in genomes])
        fitness_map = dict(results)
        for gid, genome in genomes:
            genome.fitness = fitness_map[gid]

        # Show best agent of this generation
        best = max(genomes, key=lambda g: g[1].fitness)
        best_net = neat.nn.FeedForwardNetwork.create(best[1], config)
        print(f"\n🎬 Showing gen {generation[0]} best (fitness: {best[1].fitness:.2f})")
        run_visual(best_net, generation=generation[0], stats=stats)

    winner = pop.run(eval_and_show, 50)

    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)
    print(f"\n✅ Done. Winner fitness: {winner.fitness:.2f} — saved to winner.pkl")


if __name__ == '__main__':
    if '--train' in sys.argv:
        train_and_watch()
    else:
        watch_winner()
