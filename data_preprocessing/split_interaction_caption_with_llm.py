#%%  prepare LLMs
import transformers
import torch


class LLM:
    insruction = \
"""
You are tasked with analyzing a narrative that describes the interactions between two individuals through their movements. Your goal is to identify the initiator and the receiver of the motion and to provide separate, distinct descriptions for each person's actions.
Please adhere to the following response format:
"[Initiator] The person ...
[Receiver] The person ..."
Key Guidelines:
- Refrain from using "first/second person" in your descriptions. Instead, exclusively use "the person" and "another person" when referring to the individuals involved.
- Each description should start with "The person."
- Ensure that you capture the entirety of each person's motion, including all actions in the order they occur within the interaction.
- Strictly follow the response template, and deliver precise and formal captions without any extra words.
- Limit your response to a single line.
Tip: Utilize the active voice for the initiator's actions and the passive voice for the receiver's reactions when appropriate to clearly convey the dynamics of the interaction.
"""

    query_template = \
"""
Here's the set of descriptions of the interaction:
{}
"""

    example_description_set = \
"""
One individual stands with arms crossed while another massages his right leg, and the first person softly pats the other's right arm.
One person stands with his arms crossed while the other person massages his right leg, and the first person gently pats the other person's right arm.
Two people stand facing each other, one person bends over to massage the other person's thighs, while the other person taps his shoulder.
"""
# """
# The first person walks forward and the second person blocks him by crossing their arms in front of their chest
# The first person crosses his/her arms in front of his/her chest. The second person walks towards the first person, and the first person blocks the second person's chest with his/her hands
# Two people face each other, one person walks forward, and the other person crosses his/her hands in front of his/her chest to block, then the first person stops
# """

    example_response = \
"""
[Initiator] The person bends over to massage another person's right leg.
[Receiver] The person stands with his arms crossed and is being massaged, and softly pats another person's right arm.
"""
# """
# [Initiator] The person is walking forward. [Receiver] The person is crossing their arms in front of their chest to block another person.
# [Initiator] The person walks towards another person. [Receiver] The person blocks the another person's chest with his/her hands wit his/her arms in front of his/her chest.
# [Initiator] The person walks forward [Receiver] The person crosses his/her hands in front of his/her chest to block another person.
# """

    def __init__(self, device, model_dir):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_dir,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
            use_fast=False
        )
    
    def preprocess_lines(self, lines):
        res = []
        for line in lines:
            res.append(line.replace('his/her', 'his').replace('him/her', 'him').replace('he/she', 'he'))
        return res
    
    @torch.no_grad()
    def one_round_qa(self, lines):
        lines = self.preprocess_lines(lines)
        description_set = ''
        for line in lines:
            description_set += line.strip() + '\n'
        messages = [
            {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": self.insruction + self.query_template.format(self.example_description_set)},
            {"role": "assistant", "content": self.example_response},
            {"role": "user", "content": self.query_template.format(description_set)}
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        response = outputs[0]["generated_text"][len(prompt):].strip().split('\n')

        assert response[0].startswith('[Initiator]')
        assert response[1].startswith('[Receiver]')
        return {
            'action': [response[0][len('[Initiator]'):].strip(' \n.')],
            'reaction': [response[1][len('[Receiver]'):].strip(' \n.')]
        }

#%%  Prepare src data
from pathlib import Path

data_root_dir = Path('~/Think-Then-React/data/Inter-X_Dataset').expanduser()
src_txt_path_list = (data_root_dir / 'texts').glob('*.txt')
tgt_dir = data_root_dir / 'texts_all'
tgt_dir.mkdir(exist_ok=True)

name_lines = []
for src_txt_path in src_txt_path_list:
    with src_txt_path.open('r') as f:
        lines = f.readlines()
    name_lines.append((src_txt_path.stem, lines))

#%%  Start parallel processing
import os
import torch
import json
from concurrent.futures import ProcessPoolExecutor as PPE

MODEL_DIR = os.path.expanduser('~/data/pretrained_models/llm/Meta-Llama-3-8B-Instruct')


def single_process(device_idx, name_lines_chunk, save_dir=tgt_dir, model_dir=MODEL_DIR):
    llm = LLM(device=device_idx, model_dir=model_dir)
    for i, (name, lines) in enumerate(name_lines_chunk):
        if i % 100 == 0:
            print(i)
        try:
            result = llm.one_round_qa(lines)
            result['interaction'] = lines
        except Exception as e:
            print(e)
        else:
            with (save_dir / f'{name}.json').open('w') as f:
                json.dump(result, f)

n_devices = torch.cuda.device_count()

device_list = list(range(n_devices))
name_lines_chunks = [
    name_lines[i: : n_devices] for i in range(n_devices)
]

with PPE(max_workers=n_devices) as ppe:
    list(ppe.map(single_process, device_list, name_lines_chunks))
# single_process(device_list[0], name_lines_chunks[0])

#%%  check split data
if False:
    import json
    import random
    from pathlib import Path

    data_root_dir = Path('~/data/data/motion/Inter-X_Dataset').expanduser()
    src_txt_path_list = list((data_root_dir / 'texts').glob('*.txt'))
    random.shuffle(src_txt_path_list)
    tgt_dir = data_root_dir / 'texts_action_reaction'

    for src_txt_path in src_txt_path_list[:10]:
        stem = src_txt_path.stem
        with src_txt_path.open('r') as f:
            src_lines = f.readlines()
        with (tgt_dir / f'{stem}.json').open('r') as f:
            tgt_lines = json.load(f)
        
        for src, tgt in zip(src_lines, tgt_lines):
            print(f'{src.strip()}\n{tgt}\n')
        print('-----------------------------------')
# %%
