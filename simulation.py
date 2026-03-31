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


def get_wall_sensors(agent):
    """
    4 sensors — distance to each wall, normalized 0→1.
    0 = right at the wall, 1 = far away.
    Order: N, E, S, W
    """
    ax, ay = agent
    return [
        ay / GRID_SIZE,                    # N wall (top)
        (GRID_SIZE - 1 - ax) / GRID_SIZE,  # E wall (right)
        (GRID_SIZE - 1 - ay) / GRID_SIZE,  # S wall (bottom)
        ax / GRID_SIZE,                    # W wall (left)
    ]


def get_visited_sensors(agent, visited_list):
    """
    8 directional sensors for recently visited cells.
    Returns 1.0 if a recently visited cell is nearby in that direction, 0 otherwise.
    Same cone logic as food sensors — agent can "feel" where it's already been.
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
        for vx, vy in visited_list:
            fdx = vx - ax
            fdy = vy - ay
            dist = math.sqrt(fdx**2 + fdy**2)
            if dist == 0:
                closest = 0
                break
            dot = (fdx / dist) * dx + (fdy / dist) * dy
            if dot > 0.7:
                if dist < closest:
                    closest = dist

        if closest == float('inf'):
            sensors.append(0.0)
        else:
            sensors.append(max(0.0, 1.0 - (closest / GRID_SIZE)))

    return sensors


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


def run_simulation(net, step_callback=None):
    """
    Run one agent's lifetime. Returns fitness score.

    step_callback: optional function called each step with current state dict.
    Used by the visualizer to draw frames without duplicating simulation logic.
    Signature: step_callback(state) -> bool  (return False to stop early)
    State keys: agent, food, energy, fitness, step, trail
    """
    agent = [random.randint(2, GRID_SIZE - 3), random.randint(2, GRID_SIZE - 3)]
    food = [[random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)]
            for _ in range(FOOD_COUNT)]

    fitness = 0
    energy = INITIAL_ENERGY
    recent = []    # track last 4 positions for revisit penalty
    visited = []   # last 20 positions for visited sensors
    trail = []     # last 20 positions for visualizer trail

    for step in range(STEPS):
        if energy <= 0 or not food:
            break

        # Get sensor inputs (8 food + 8 visited + 4 walls = 20 total)
        inputs = get_sensors(agent, food) + get_visited_sensors(agent, visited) + get_wall_sensors(agent)

        # Brain decides
        output = net.activate(inputs)
        action = output.index(max(output))  # 0=N, 1=E, 2=S, 3=W

        # Move
        moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = moves[action]
        old_x, old_y = agent[0], agent[1]
        agent[0] = max(0, min(GRID_SIZE - 1, agent[0] + dx))
        agent[1] = max(0, min(GRID_SIZE - 1, agent[1] + dy))

        # Punish hitting the wall
        if agent[0] == old_x and agent[1] == old_y and (dx != 0 or dy != 0):
            fitness -= 0.2

        energy -= MOVE_COST

        # Punish revisiting a square seen in the last 4 steps
        pos = tuple(agent)
        if pos in recent:
            fitness -= 0.2
        recent.append(pos)
        if len(recent) > 4:
            recent.pop(0)

        # Update trail for visualizer and visited sensors
        trail.append((agent[0], agent[1]))
        if len(trail) > 20:
            trail.pop(0)
        visited.append((agent[0], agent[1]))
        if len(visited) > 20:
            visited.pop(0)

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

        # Call visualizer callback if provided
        if step_callback is not None:
            state = {
                'agent': agent[:],
                'food': [f[:] for f in food],
                'energy': energy,
                'fitness': fitness,
                'step': step,
                'trail': list(trail),
            }
            keep_going = step_callback(state)
            if keep_going is False:
                break

    # Penalty for dying early
    if energy <= 0:
        fitness -= 5

    return fitness
