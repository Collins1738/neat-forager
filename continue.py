"""
continue.py — Resume training seeded from winner.pkl
The entire starting population is built from mutated copies of the winner.
"""

import neat
import pickle
import copy
import os
from multiprocessing import Pool
from simulation import run_simulation

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'forager_config.txt')


def eval_genome(args):
    genome_id, genome, config = args
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    scores = [run_simulation(net) for _ in range(3)]
    return genome_id, sum(scores) / len(scores)


def eval_genomes(genomes, config):
    with Pool() as pool:
        results = pool.map(eval_genome, [(gid, g, config) for gid, g in genomes])
    fitness_map = dict(results)
    for gid, genome in genomes:
        genome.fitness = fitness_map[gid]


def eval_and_show(generation_counter):
    """Returns an eval function that shows the best agent each gen."""
    from visualize import run_visual

    def _eval(genomes, config):
        generation_counter[0] += 1

        with Pool() as pool:
            results = pool.map(eval_genome, [(gid, g, config) for gid, g in genomes])
        fitness_map = dict(results)
        for gid, genome in genomes:
            genome.fitness = fitness_map[gid]

        best = max(genomes, key=lambda g: g[1].fitness)
        best_net = neat.nn.FeedForwardNetwork.create(best[1], config)
        print(f"\n🎬 Showing gen {generation_counter[0]} best (fitness: {best[1].fitness:.2f})")
        run_visual(best_net, generation=generation_counter[0])

    return _eval


def seed_population_from_winner(pop, winner, config):
    """Replace the starting population with mutated copies of the winner."""
    new_population = {}
    for genome_id in pop.population:
        child = copy.deepcopy(winner)
        child.key = genome_id
        child.fitness = None
        child.mutate(config.genome_config)
        new_population[genome_id] = child

    pop.population = new_population
    pop.species.speciate(config, pop.population, pop.generation)


def run(generations=200, visual=False):
    if not os.path.exists('winner.pkl'):
        print("❌ No winner.pkl found. Run train.py first.")
        return

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    with open('winner.pkl', 'rb') as f:
        winner = pickle.load(f)

    print(f"📂 Loaded winner (fitness: {winner.fitness:.2f})")
    print(f"🧬 Seeding population of {config.pop_size} from winner...")

    pop = neat.Population(config)
    seed_population_from_winner(pop, winner, config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    print(f"🚀 Continuing evolution for {generations} more generations...\n")

    if visual:
        generation_counter = [0]
        new_winner = pop.run(eval_and_show(generation_counter), generations)
    else:
        new_winner = pop.run(eval_genomes, generations)

    print(f"\n✅ New best fitness: {new_winner.fitness:.2f}")

    with open('winner.pkl', 'wb') as f:
        pickle.dump(new_winner, f)
    print("💾 Saved new winner to winner.pkl")


if __name__ == '__main__':
    import sys
    visual = '--visual' in sys.argv
    run(visual=visual)
