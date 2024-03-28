import collections
import functools
import pathlib

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageEnhance


class AttrDict(dict):
    __getattr__ = dict.__getitem__


class staticproperty:

    def __init__(self, function):
        self.function = function
    
    def __get__(self, instance, owner=None):
        return self.function()


class World:

    def __init__(self, area, materials, chunk_size, el_vars, el_freq, total_dreamer, el_app_freq):
        self.area = area
        self._chunk_size = chunk_size
        self._mat_names = {i: x for i, x in enumerate([None] + materials)}
        self._mat_names_vars = {i: x for i, x in enumerate([None] + materials)}
        self._mat_ids = {x: i for i, x in enumerate ([None] + materials)}
        self._mat_ids_vars = {x: i for i, x in enumerate ([None] + materials)}

        # self.n_reset = 0
        # self.n_tree = 0
        # self.n_coal = 0
        # self.n_cow = 0
        # self.n_zombie = 0
        # self.n_skeleton = 0

        # self.avg_tree = 0
        # self.avg_coal = 0
        # self.avg_cow = 0
        # self.avg_zombie = 0
        # self.avg_skeleton = 0

        # self.min_tree = 1000
        # self.min_coal = 1000
        # self.min_cow = 1000
        # self.min_zombie = 1000
        # self.min_skeleton = 1000

        # self.max_tree = 0
        # self.max_coal = 0
        # self.max_cow = 0
        # self.max_zombie = 0
        # self.max_skeleton = 0

        self.reset()
        # el_vars = 'atwczsuki' (order does not matter)
        self.el_vars = el_vars
        self.el_vars_keys = {
            'player': 'p',
            'player-sleep': 'p',
            'player-left': 'p',
            'player-right': 'p',
            'player-up': 'p',
            'player-down': 'p',
            'tree': 't',
            'water': 'w',
            'cow': 'c',
            'zombie': 'z',
            'stone': 's',
            'coal': 'u',
            'skeleton': 'k',
            'iron': 'i',
        }
        # el_freq = '100,0,0,0' <- default
        # Examples:
        # v0 - 100,0,0,0 <- eval 1
        # v1 - 50,50,0,0 <- eval 2
        # v2 - 70,30,0,0
        # v3 - 90,10,0,0
        # v4 - 33,33,33,0 <- eval 3
        # v5 - 50,25,25,0
        # v6 - 80,10,10,0
        # v7 - 25,25,25,25 <- eval 4
        # v8 - 40,20,20,20
        # v9 - 70,10,10,10
        # v10 - 48,24,16,12
        self.el_freq = np.cumsum([int(e) for e in el_freq.split(',')])
        self.total_dreamer = total_dreamer
        if el_app_freq is None:
            el_app_freq = 'sssss'
        else:
            if el_app_freq == 'easyX2':
                el_app_freq = 'dddhh'
            elif el_app_freq == 'easyX4':
                el_app_freq = 'fffqq'
            elif el_app_freq == 'default':
                el_app_freq = 'sssss'
            elif el_app_freq == 'mix':
                el_app_freq = 'fffff'
            elif el_app_freq == 'hardX2':
                el_app_freq = 'hhhdd'
            elif el_app_freq == 'hardX4':
                el_app_freq = 'qqqff'

        self.el_app_freq = el_app_freq
        freq_dict_tree = {
            'q': 0.945, # 50
            'h': 0.9, # 95
            's': 0.8, # 190
            'd': 0.6, # 380
            'f': 0.2, # 760
        }
        freq_dict_coal = {
            'q': 0.963, # 12
            'h': 0.92, # 25
            's': 0.85, # 50
            'd': 0.7, # 100
            'f': 0.4, # 200
        }
        freq_dict_cow = {
            'q': 0.9968, # 6
            'h': 0.993, # 13
            's': 0.985, # 26
            'd': 0.97, # 52
            'f': 0.91, # 104
        }
        freq_dict_zombie = {
            'q': 0.998, # 4
            'h': 0.9968, # 7
            's': 0.993, # 14.6
            'd': 0.985, # 30
            'f': 0.96, # 60
        }
        freq_dict_skeleton = {
            'q': 0.99, # 2
            'h': 0.977, # 4
            's': 0.95, # 9.5
            'd': 0.9, # 19
            'f': 0.8, # 38
        }
        self.freq_tree = freq_dict_tree[el_app_freq[0]]  # el_app_freq['tree']
        self.freq_coal = freq_dict_coal[el_app_freq[1]]  # el_app_freq['coal']
        self.freq_cow = freq_dict_cow[el_app_freq[2]]  # el_app_freq['cow']
        self.freq_zombie = freq_dict_zombie[el_app_freq[3]]  # el_app_freq['zombie']
        self.freq_skeleton = freq_dict_skeleton[el_app_freq[4]]  # el_app_freq['skeleton']

    def min_max(self, min, max, val):
        if min > val and val > 0:
            min = val
        if max < val:
            max = val
        return min, max
    
    def reset(self, seed=None):
        # self.n_reset += 1
        # self.avg_tree = (self.n_reset - 1) / self.n_reset * self.avg_tree + self.n_tree / self.n_reset
        # self.avg_coal = (self.n_reset - 1) / self.n_reset * self.avg_coal + self.n_coal / self.n_reset
        # self.avg_cow = (self.n_reset - 1) / self.n_reset * self.avg_cow + self.n_cow / self.n_reset
        # self.avg_zombie = (self.n_reset - 1) / self.n_reset * self.avg_zombie + self.n_zombie / self.n_reset
        # self.avg_skeleton = (self.n_reset - 1) / self.n_reset * self.avg_skeleton + self.n_skeleton / self.n_reset

        # self.min_tree, self.max_tree = self.min_max(self.min_tree, self.max_tree, self.n_tree)
        # self.min_coal, self.max_coal = self.min_max(self.min_coal, self.max_coal, self.n_coal)
        # self.min_cow, self.max_cow = self.min_max(self.min_cow, self.max_cow, self.n_cow)
        # self.min_zombie, self.max_zombie = self.min_max(self.min_zombie, self.max_zombie, self.n_zombie)
        # self.min_skeleton, self.max_skeleton = self.min_max(self.min_skeleton, self.max_skeleton, self.n_skeleton)

        # print('n', self.n_reset)
        # print('tree    :{:.1f},{:.1f},{:.1f},{:.1f}'.format(self.n_tree, self.avg_tree, self.min_tree, self.max_tree))
        # print('coal    :{:.1f},{:.1f},{:.1f},{:.1f}'.format(self.n_coal, self.avg_coal, self.min_coal, self.max_coal))
        # print('cow     :{:.1f},{:.1f},{:.1f},{:.1f}'.format(self.n_cow, self.avg_cow, self.min_cow, self.max_cow))
        # print('zombie  :{:.1f},{:.1f},{:.1f},{:.1f}'.format(self.n_zombie, self.avg_zombie, self.min_zombie, self.max_zombie))
        # print('skeleton:{:.1f},{:.1f},{:.1f},{:.1f}'.format(self.n_skeleton, self.avg_skeleton, self.min_skeleton, self.max_skeleton))

        # self.n_tree = 0
        # self.n_coal = 0
        # self.n_cow = 0
        # self.n_zombie = 0
        # self.n_skeleton = 0

        self.random = np.random.RandomState(seed)
        self.daylight = 0.0
        self._chunks = collections.defaultdict(set)
        self._objects = [None]
        self._mat_map = np.zeros(self.area, np.uint8)
        self._mat_map_vars = np.zeros(self.area, np.uint8)
        self._obj_map = np.zeros(self.area, np.uint32)

    @property
    def objects(self):
        # Return a new list so the objects cannot change while being iterated over.
        return [obj for obj in self._objects if obj]
    
    @property
    def chunks(self):
        return self._chunks.copy()
    
    def add(self, obj):
        assert hasattr(obj, 'pos')
        obj.pos = np.array(obj.pos)
        assert self._obj_map[tuple(obj.pos)] == 0
        index = len(self._objects)
        self._objects.append(obj)
        self._obj_map[tuple(obj.pos)] = index
        self._chunks[self.chunk_key(obj.pos)].add(obj)
    
    def remove(self, obj):
        if obj.removed:
            return
        self._objects[self._obj_map[tuple(obj.pos)]] = None
        self._obj_map[tuple(obj.pos)] = 0
        self._chunks[self.chunk_key(obj.pos)].remove(obj)
        obj.removed=True
    
    def move(self, obj, pos):
        if obj.removed:
            return
        pos = np.array(pos)
        assert self._obj_map[tuple(pos)] == 0
        index = self._obj_map[tuple(obj.pos)]
        self._obj_map[tuple(pos)] = index
        self._obj_map[tuple(obj.pos)] = 0
        old_chunk = self.chunk_key(obj.pos)
        new_chunk = self.chunk_key(pos)
        if old_chunk != new_chunk:
            self._chunks[old_chunk].remove(obj)
            self._chunks[new_chunk].add(obj)
        obj.pos = pos
    
    def get_el_var(self, name):
        # Sample a particular variant of the element (if required)
        if name not in self.el_vars_keys:
            return ''
        el_id = self.el_vars_keys[name]
        if el_id not in self.el_vars:
            return ''
        else:
            sample = self.random.uniform() * 100
            idx = next(x[0] for x in enumerate(self.el_freq) if sample <= x[1])
            return str(idx)

    def __setitem__(self, pos, material):
        if material not in self._mat_ids:
            id_ = len(self._mat_ids)
            self._mat_ids[material] = id_
        self._mat_map[tuple(pos)] = self._mat_ids[material]
        if self.total_dreamer:
            material_var = material
        else:
            material_var = material + self.get_el_var(material)
        if material_var not in self._mat_ids_vars:
            id_ = len(self._mat_ids_vars)
            self._mat_ids_vars[material_var] = id_
            self._mat_names_vars[id_] = material_var
        self._mat_map_vars[tuple(pos)] = self._mat_ids_vars[material_var]

    def __getitem__(self, pos):
        if not _inside((0, 0), pos, self.area):
            return None, None, None
        material = self._mat_names[self._mat_map[tuple(pos)]]
        material_vars = self._mat_names_vars[self._mat_map_vars[tuple(pos)]]
        obj = self._objects[self._obj_map[tuple(pos)]]
        return material, obj, material_vars
    
    def nearby(self, pos, distance):
        (x, y), d = pos, distance
        ids = set(self._mat_map[x - d: x + d + 1, y - d: y + d + 1].flatten().tolist())
        materials = tuple(self._mat_names[x] for x in ids)
        indices = self._obj_map[x - d: x + d + 1, y - d: y + d + 1].flatten().tolist()
        objs = {self._objects[i] for i in indices if i > 0}
        return materials, objs
    
    def mask(self, xmin, xmax, ymin, ymax, material):
        region = self._mat_map[xmin: xmax, ymin: ymax]
        return (region == self._mat_ids[material])

    def count(self, material):
        return (self._mat_map == self._mat_ids[material]).sum()
    
    def chunk_key(self, pos):
        (x, y), (csx, csy) = pos, self._chunk_size
        xmin, ymin = (x // csx) * csx, (y // csy) * csy
        xmax = min(xmin + csx, self.area[0])
        ymax = min(ymin + csy, self.area[1])
        return (xmin, xmax, ymin, ymax)


class Textures:

    def __init__(self, directory, world):
        self._originals = {}
        self._textures = {}
        for filename in pathlib.Path(directory).glob('*.png'):
            image = imageio.imread(filename.read_bytes())
            image = image.transpose((1, 0) + tuple(range(2, len(image.shape))))
            self._originals[filename.stem] = image
            self._textures[(filename.stem, image.shape[:2])] = image
        self.world = world
    
    def get(self, name, size):
        if name is None:
            name = 'unknown'
        if self.world.total_dreamer:
            name = name + self.world.get_el_var(name)
        size = int(size[0]), int(size[1])
        key = name, size
        if key not in self._textures:
            image = self._originals[name]
            image = Image.fromarray(image)
            image = image.resize(size[::-1], resample=Image.NEAREST)
            image = np.array(image)
            self._textures[key] = image
        return self._textures[key]
        

class LocalView:

    def __init__(self, world, textures, grid):
        self._world = world
        self._textures = textures
        self._grid = np.array(grid)
        self._offset = self._grid // 2
        self._area = np.array(self._world.area)
        self._center = None

    def __call__(self, player, unit):
        self._unit = np.array(unit)
        self._center = np.array(player.pos)
        canvas = np.zeros(tuple(self._grid * unit) + (3,), np.uint8) + 127
        for x in range(self._grid[0]):
            for y in range(self._grid[1]):
                pos = self._center + np.array([x, y]) - self._offset
                if not _inside((0, 0), pos, self._area):
                    continue
                texture = self._textures.get(self._world[pos][2], unit)
                _draw(canvas, np.array([x, y]) * unit, texture)
        for obj in self._world.objects:
            pos = obj.pos - self._center + self._offset
            if not _inside((0, 0), pos, self._grid):
                continue
            if self._world.total_dreamer:
                texture = self._textures.get(obj.texture, unit)
            else:
                texture = self._textures.get(obj.texture_vars, unit)
            _draw_alpha(canvas, pos * unit, texture)
        canvas = self._light(canvas, self._world.daylight)
        if player.sleeping:
            canvas = self._sleep(canvas)
        # if player.health < 1:
        #     canvas = self._tint(canvas, (128, 0, 0), 0.6)
        return canvas
    
    def _light(self, canvas, daylight):
        night = canvas
        if daylight < 0.5:
            night = self._noise(night, 2 * (0.5 - daylight), 0.5)
        night = np.array(ImageEnhance.Color(Image.fromarray(night.astype(np.uint8))).enhance(0.4))
        night = self._tint(night, (0, 16, 64), 0.5)
        return daylight * canvas + (1 - daylight) * night
    
    def _sleep(self, canvas):
        canvas = np.array(ImageEnhance.Color(Image.fromarray(canvas.astype(np.uint8))).enhance(0.0))
        canvas = self._tint(canvas, (0, 0, 16), 0.5)
        return canvas
    
    def _tint(self, canvas, color, amount):
        color = np.array(color)
        return (1 - amount) * canvas + amount * color

    def _noise(self, canvas, amount, stddev):
        noise = self._world.random.uniform(32, 127, canvas.shape[:2])[..., None]
        mask = amount * self._vignette(canvas.shape, stddev)[..., None]
        return (1 - mask) * canvas + mask * noise

    @functools.lru_cache(10)
    def _vignette(self, shape, stddev):
        xs, ys = np.meshgrid(
            np.linspace(-1, 1, shape[0]),
            np.linspace(-1, 1, shape[1]))
        return 1 - np.exp(-0.5 * (xs ** 2 + ys ** 2) / (stddev ** 2)).T


class ItemView:

    def __init__(self, textures, grid, render_scoreboard=True):
        self._textures = textures
        self._grid = np.array(grid)
        self._render_scoreboard = render_scoreboard
    
    def __call__(self, inventory, unit):
        unit = np.array(unit)
        canvas = np.zeros(tuple(self._grid * unit) + (3,), np.uint8)
        for index, (item, amount) in enumerate(inventory.items()):
            if amount < 1:
                continue
            self._item(canvas, index, item, unit)
            self._amount(canvas, index, amount, unit)
        return canvas
    
    def _item(self, canvas, index, item, unit):
        pos = index % self._grid[0], index // self._grid[0]
        pos = (pos * unit + 0.1 * unit).astype(np.int32)
        texture = self._textures.get(item, 0.8 * unit)
        if self._render_scoreboard:
            _draw_alpha(canvas, pos, texture)
    
    def _amount(self, canvas, index, amount, unit):
        pos = index % self._grid[0], index // self._grid[0]
        pos = (pos * unit + 0.4 * unit).astype(np.int32)
        text = str(amount) if amount in list(range(10)) else 'unknown'
        texture = self._textures.get(text, 0.6 * unit)
        if self._render_scoreboard:
            _draw_alpha(canvas, pos, texture)


class SemanticView:

    def __init__(self, world, obj_types):
        self._world = world
        self._mat_ids = world._mat_ids.copy()
        self._obj_ids = {c: len(self._mat_ids) + i for i, c in enumerate(obj_types)}
    
    def __call__(self):
        canvas = self._world._mat_map.copy()
        for obj in self._world.objects:
            canvas[tuple(obj.pos)] = self._obj_ids[type(obj)]
        return canvas


def _inside(lhs, mid, rhs):
    return (lhs[0] <= mid[0] < rhs[0]) and (lhs[1] <= mid[1] < rhs[1])


def _draw(canvas, pos, texture):
    (x, y), (w, h) = pos, texture.shape[:2]
    if texture.shape[-1] == 4:
        texture = texture[..., :3]
    canvas[x: x+ w, y: y + h] = texture

def _draw_alpha(canvas, pos, texture):
    (x, y), (w, h) = pos, texture.shape[:2]
    if texture.shape[-1] == 4:
        alpha = texture[..., 3:].astype(np.float32) / 255
        texture = texture[..., :3].astype(np.float32) / 255
        current = canvas[x: x + w, y: y + h].astype(np.float32) / 255
        blended = alpha * texture + (1 - alpha) * current
        texture = (255 * blended).astype(np.uint8)
    canvas[x: x + w, y: y + h] = texture
