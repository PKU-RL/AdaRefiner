import numpy as np
import torch
import torch.nn.functional as F

API_KEY = '' # Your OpenAI API key
BASE_URL = 'https://api.openai.com/v1'
LLM_PATH = '' # Path to the LLM model (Llama-2-7b-chat-hf)
GPT_MODEL = 'gpt-4' # GPT model name
QUERY_INTERVAL = 100 # Query interval for LLMs
DATA_PATH = './data/' # Data path for SFT

TYPE_DICT = {
    1: 'water',
    2: 'grass',
    3: 'stone',
    4: 'path',
    5: 'sand',
    6: 'tree',
    7: 'lava',
    8: 'coal',
    9: 'iron',
    10: 'diamond',
    11: 'table',
    12: 'furnace',
    13: 'player',
    14: 'cow',
    15: 'zombie',
    16: 'skeleton',
    17: 'arrow',
    18: 'plant'
}

ACTIONS_NAME = [
    'noop',
    'move_left',
    'move_right',
    'move_up',
    'move_down',
    'do',
    'sleep',
    'place_stone',
    'place_table',
    'place_furnace',
    'place_plant',
    'make_wood_pickaxe',
    'make_stone_pickaxe',
    'make_iron_pickaxe',
    'make_wood_sword',
    'make_stone_sword',
    'make_iron_sword'
  ]

def l_func(x):
    return 1 / (1 + torch.exp(-10 * (x - 0.1)))

def compute_l_score(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    result = l_func(torch.mm(a_norm, b_norm.transpose(0, 1)))
    
    return result[0].detach().cpu().numpy()

def compute_bin_score(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    result = l_func(torch.mm(a_norm, b_norm.transpose(0, 1)))[0].detach().cpu().numpy()
    
    if result >= 0.5:
        return 1
    else:
        return 0

def get_fov_types(info):
    pos = info['player_pos']
    obs = info['semantic']

    fov_size = np.array([9, 7])
    top_left = np.maximum(pos - fov_size // 2, 0)
    bottom_right = np.minimum(pos + fov_size // 2 + 1, obs.shape)
    fov = obs[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    pad_top = top_left[0] - pos[0] + fov_size[0] // 2
    pad_bottom = pos[0] + fov_size[0] // 2 + 1 - bottom_right[0]
    pad_left = top_left[1] - pos[1] + fov_size[1] // 2
    pad_right = pos[1] + fov_size[1] // 2 + 1 - bottom_right[1]
    fov = np.pad(fov, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    types = np.unique(fov)
    type_strings = [TYPE_DICT[t] for t in types if t != 13 and t != 0]
    
    return type_strings

def llm_template(fov, status, a_last, inst, l_score):
    llm_system_message = """You are a professional game analyst. A player is playing a 2D Minecraft game. You will get the player's observation, status information, and its comprehension score of language guidance. You will be asked to provide concise summaries and suggestions about this player."""
    llm_user_message = f"Player sees: <{fov}>. Player status: <{status}>. Past action: <{a_last}>. Past sub-goals: <{inst}>. Player's comprehension score: <{l_score:.3f}>."
    llm_prompt = f"[INST] <<SYS>>\n{llm_system_message}\n<</SYS>>\n\n{llm_user_message} [/INST]"
    
    return llm_prompt

def gpt_template(fov, status, a_last, inst, adapted_prompt):
    gpt_system_message = """You are a professional game analyst. A player is playing a 2D Minecraft game. You will also get analysis about this player from another analyst. Available sub-goals are: <eat plant, attack zombie, attack skeleton, attack cow, chop tree, mine stone, mine coal, mine iron, mine diamond, drink water, chop grass, sleep, place stone, place crafting table, place furnace, place plant, make wood pickaxe, make stone pickaxe, make iron pickaxe, make wood sword, make stone sword, make iron sword, plant row, chop grass with wood pickaxe, vegetarianism, make workshop, survival, deforestation, work and sleep, gardening, wilderness survival>. You are asked to suggest 3 sub-goals from available sub-goals in one sentence without any other words."""
    gpt_user_message = f"Player sees: <{fov}>. Player status: <{status}>. Past action: <{a_last}>. Past sub-goals: <{inst}>. Analysis: <{adapted_prompt}>."
    gpt_prompt = [
        {"role": "system", "content": gpt_system_message},
        {"role": "user", "content": gpt_user_message},
    ]

    return gpt_prompt