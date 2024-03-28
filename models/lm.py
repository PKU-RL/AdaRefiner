import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from utils import get_fov_types, compute_l_score, llm_template, gpt_template, API_KEY, BASE_URL, LLM_PATH, GPT_MODEL, DATA_PATH

class AdaLM:

    def __init__(self):

        # set gpt
        self.gpt_client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        
        # load llm
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, "float16"),
            bnb_4bit_use_double_quant=False,
        )
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_PATH,
            quantization_config=bnb_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
        self.text_enc = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    
    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=1, max=60))
    def gpt_generate(self, prompt, model=GPT_MODEL):
        response = self.gpt_client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=1,
        )
        return response.choices[0].message.content

    def llm_generate(self, prompt):
        num_new_tokens = 100
        num_prompt_tokens = len(self.tokenizer(prompt)['input_ids'])
        max_length = num_prompt_tokens + num_new_tokens
        gen = pipeline('text-generation', model=self.llm_model, tokenizer=self.tokenizer, max_length=max_length, return_full_text=False)
        result = gen(prompt)
        return result[0]['generated_text'].strip()
    
    def inst_generate(self, info, a_hist_list, inst_last):
        a_hist = ', '.join(a_hist_list)
        a_last = a_hist_list[-1] if a_hist_list else ''
        a_emb = self.text_enc.encode(a_hist)
        inst_emb = self.text_enc.encode(inst_last)
        l_score = compute_l_score(a_emb, inst_emb)[0]
        fov = ', '.join(get_fov_types(info))
        status = ', '.join([f"{v} {k}" for k, v in info['inventory'].items() if v > 0])
        
        # llm generate
        llm_prompt = llm_template(fov, status, a_last, inst_last, l_score)
        adapted_prompt = self.llm_generate(llm_prompt)

        # gpt generate
        gpt_prompt = gpt_template(fov, status, a_last, inst_last, adapted_prompt)
        inst = self.gpt_generate(gpt_prompt)
        
        data = {
            "prompt": llm_prompt,
            "response": adapted_prompt,
        }
        with open(f'{DATA_PATH}/data.jsonl', 'a') as f:
            f.write(json.dumps(data) + '\n')
        
        return inst