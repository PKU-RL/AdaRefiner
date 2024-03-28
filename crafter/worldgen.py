import functools

import numpy as np
import opensimplex

from . import constants
from . import objects


def generate_world(world, player):
    simplex = opensimplex.OpenSimplex(seed=world.random.randint(0, 2 ** 31 - 1))
    tunnels = np.zeros(world.area, np.bool_)
    for x in range(world.area[0]):
        for y in range(world.area[1]):
            _set_material(world, (x, y), player, tunnels, simplex)
    for x in range(world.area[0]):
        for y in range(world.area[1]):
            _set_object(world, (x, y), player, tunnels)


def _set_material(world, pos, player, tunnels, simplex):
    x, y = pos
    simplex = functools.partial(_simplex, simplex)
    uniform = world.random.uniform
    start = 4 - np.sqrt((x - player.pos[0]) ** 2 + (y - player.pos[1]) ** 2)
    start += 2 * simplex(x, y, 8, 3)
    start = 1 / (1 + np.exp(-start))
    water = simplex(x, y, 3, {15: 1, 5: 0.15}, False) + 0.1
    water -= 2 * start
    mountain = simplex(x, y, 0, {15: 1, 5: 0.3})
    mountain -= 4 * start + 0.3 * water
    if start > 0.5:
        world[x, y] = 'grass'
    elif mountain > 0.15:
        if (simplex(x, y, 6, 7) > 0.15 and mountain > 0.3):  # cave
            world[x, y] = 'path'
        elif simplex(2 * x, y / 5, 7, 3) > 0.4:  # horizonal tunnel
            world[x, y] = 'path'
            tunnels[x, y] = True
        elif simplex(x / 5, 2 * y, 7, 3) > 0.4:  # vertical tunnel
            world[x, y] = 'path'
            tunnels[x, y] = True
        elif simplex(x, y, 1, 8) > 0 and uniform() > world.freq_coal:
            world[x, y] = 'coal'  # TODO: freq_app
            # world.n_coal += 1
        elif simplex(x, y, 2, 6) > 0.4 and uniform() > 0.75:
            world[x, y] = 'iron'
        elif mountain > 0.18 and uniform() > 0.994:
            world[x, y] = 'diamond'
        elif mountain > 0.3 and simplex(x, y, 6, 5) > 0.35:
            world[x, y] = 'lava'
        else:
            world[x, y] = 'stone'
    elif 0.25 < water <= 0.35 and simplex(x, y, 4, 9) > -0.2:
        world[x, y] = 'sand'
    elif 0.3 < water:
        world[x, y] = 'water'
    else:  # grassland
        if simplex(x, y, 5, 7) > 0 and uniform() > world.freq_tree:
            world[x, y] = 'tree'  # TODO: freq_app
            # world.n_tree += 1
        else:
            world[x, y] = 'grass'


def _set_object(world, pos, player, tunnels):
    x, y = pos
    uniform = world.random.uniform
    dist = np.sqrt((x - player.pos[0]) ** 2 + (y - player.pos[1]) ** 2)
    material, _, _ = world[x, y]
    if material not in constants.walkable:
        pass
    elif dist > 3 and material == 'grass' and uniform() > world.freq_cow:
        world.add(objects.Cow(world, (x, y)))  # TODO: freq_app
        # world.n_cow += 1
    elif dist > 10 and uniform() > world.freq_zombie:
        world.add(objects.Zombie(world, (x, y), player))  # TODO: freq_app
        # world.n_zombie += 1
    elif material == 'path' and tunnels[x, y] and uniform() > world.freq_skeleton:
        world.add(objects.Skeleton(world, (x, y), player))  # TODO: freq_app
        # world.n_skeleton += 1
    

def _simplex(simplex, x, y, z, sizes, normalize=True):
    if not isinstance(sizes, dict):
        sizes = {sizes: 1}
    value = 0
    for size, weight in sizes.items():
        if hasattr(simplex, 'noise3d'):
            noise = simplex.noise3d(x / size, y / size, z)
        else:
            noise = simplex.noise3(x / size, y / size, z)
        value += weight * noise
    if normalize:
        value /= sum(sizes.values())
    return value