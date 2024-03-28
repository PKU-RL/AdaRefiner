import torch
import crafter
import framework
from config.default import register_args
import stable_baselines3 as sb3
from utils import get_fov_types, compute_l_score, llm_template, gpt_template, ACTIONS_NAME
from collections import deque

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer

from openai import OpenAI, AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from utils import API_KEY, BASE_URL

@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=1, max=60))
def gpt_generate(prompt, client, model="gpt-4"):
    response = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=1,
    )
    return response.choices[0].message.content

def llm_generate(prompt, model, tokenizer):
    num_new_tokens = 100
    num_prompt_tokens = len(tokenizer(prompt)['input_ids'])
    max_length = num_prompt_tokens + num_new_tokens
    gen = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=max_length, return_full_text=False)
    result = gen(prompt)
    return result[0]['generated_text'].strip()

def main():
    # load env
    helper = framework.helpers.TrainingHelper(register_args=register_args)
    env = crafter.Env()
    env = crafter.Recorder(
        env,
        './results',
        helper,
        save_stats=True,
        save_video=True,
        save_episode=False,
        log_every_n_episodes=1,
    )

    # set gpt
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    # load policy
    model = sb3.PPO(helper, env)
    ckpt = torch.load('./model_files/policy.pth', map_location=torch.device('cuda:0'))
    model.policy.load_state_dict(ckpt['model_state_dict'])

    # load llm
    llm_path = "./model_files/llama-2-7b-crafter"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, "float16"),
        bnb_4bit_use_double_quant=False,
    )
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    text_enc = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    query_interval = 10

    obs = env.reset()
    done = False
    steps = 0
    rew = 0
    a_hist_list = deque(maxlen=query_interval)
    inst = ''
    fov = ''
    status = ''
    
    while not done:
        a_hist = ', '.join(a_hist_list)
        a_last = a_hist_list[-1] if a_hist_list else ''
        a_emb = text_enc.encode(a_hist)
        inst_emb = text_enc.encode(inst)
        l_score = compute_l_score(a_emb, inst_emb)[0]
        
        if steps % query_interval == 0:
            # llm generate
            llm_prompt = llm_template(fov, status, a_last, inst, l_score)
            adapted_prompt = llm_generate(llm_prompt, llm_model, tokenizer)
        
            # gpt generate
            gpt_prompt = gpt_template(fov, status, a_last, inst, adapted_prompt)
            inst = gpt_generate(gpt_prompt, client)

        action, _, _ = model.policy.predict(obs, instruction=inst)
        a_hist_list.append(ACTIONS_NAME[action])
        obs, reward, done, info = env.step(action)
        fov = ', '.join(get_fov_types(info))
        status = ', '.join([f"{v} {k}" for k, v in info['inventory'].items() if v > 0])
        
        rew += reward
        steps += 1
        print(f"Current step: {steps}. Reward: {reward}")
    
    print(f"Episode reward: {rew}")
    helper.finish()

if __name__ == '__main__':
    main()