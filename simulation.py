"""
NEAT Forager Simulation
Agents learn to find food and survive over generations.
"""

import random
import math

GRID_SIZE = 30
FOOD_COUNT = 15
STEPS = 300
INITIAL_ENERGY = 150
MOVE_COST = 1
EAT_GAIN = 40


def get_sensors(agent, food_list):
    """
    8 directional sensors — each returns distance to nearest food in that direction.
    Directions: N, NE, E, SE, S, SW, W, NW
    Returns normalized values (0 = far/nothing, 1 = right on top of it)
    """
    directions = [
        (0, -1),   # N
        (1, -1),   # NE
        (1, 0),    # E
        (1, 1),    # SE
        (0, 1),    # S
        (-1, 1),   # SW
        (-1, 0),   # W
        (-1, -1),  # NW
    ]

    sensors = []
    ax, ay = agent

    for dx, dy in directions:
        closest = float('inf')
        for fx, fy in food_list:
            # Check if food is roughly in this direction
            fdx = fx - ax
            fdy = fy - ay
            dist = math.sqrt(fdx**2 + fdy**2)
            if dist == 0:
                closest = 0
                break
            # dot product to check alignment with direction
            dot = (fdx / dist) * dx + (fdy / dist) * dy
            if dot > 0.7:  # within ~45 degrees of this direction
                if dist < closest:
                    closest = dist

        # Normalize: close = high value, far/none = 0
        if closest == float('inf'):
            sensors.append(0.0)
        else:
            sensors.append(max(0.0, 1.0 - (closest / GRID_SIZE)))

    return sensors


def run_simulation(net):
    """Run one agent's lifetime. Returns fitness score."""
    agent = [random.randint(2, GRID_SIZE - 3), random.randint(2, GRID_SIZE - 3)]
    food = [[random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)]
            for _ in range(FOOD_COUNT)]

    fitness = 0
    energy = INITIAL_ENERGY
    recent = []  # track last 4 positions for revisit penalty

    for step in range(STEPS):
        if energy <= 0 or not food:
            break

        # Get sensor inputs
        inputs = get_sensors(agent, food)

        # Brain decides
        output = net.activate(inputs)
        action = output.index(max(output))  # 0=N, 1=E, 2=S, 3=W

        # Move
        moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = moves[action]
        agent[0] = max(0, min(GRID_SIZE - 1, agent[0] + dx))
        agent[1] = max(0, min(GRID_SIZE - 1, agent[1] + dy))

        energy -= MOVE_COST

        # Punish revisiting a square seen in the last 4 steps
        pos = tuple(agent)
        if pos in recent:
            fitness -= 0.5
        recent.append(pos)
        if len(recent) > 4:
            recent.pop(0)

        # Check for food
        for f in food[:]:
            if agent[0] == f[0] and agent[1] == f[1]:
                food.remove(f)
                fitness += 10
                energy = min(INITIAL_ENERGY, energy + EAT_GAIN)
                # Respawn food elsewhere
                food.append([random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)])

        # Bonus for surviving longer
        fitness += 0.01

    # Penalty for dying early
    if energy <= 0:
        fitness -= 5

    return max(0, fitness)
