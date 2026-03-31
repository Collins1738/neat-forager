"""
train.py — Run NEAT evolution without visuals.
Good for fast training. Saves the winner genome to winner.pkl
"""

import neat
import pickle
import os
from simulation import run_simulation

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'forager_config.txt')


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        scores = [run_simulation(net) for _ in range(3)]
        genome.fitness = sum(scores) / len(scores)


def run(generations=50):
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

    winner = pop.run(eval_genomes, generations)

    print(f"\n✅ Best genome fitness: {winner.fitness:.2f}")

    # Save winner
    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)
    print("💾 Winner saved to winner.pkl")

    return winner, config


if __name__ == '__main__':
    run()
