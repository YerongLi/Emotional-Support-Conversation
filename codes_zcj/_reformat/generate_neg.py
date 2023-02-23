import json
import tqdm
import numpy as np
import multiprocessing as mp
import nltk
import random
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
def generate(instruction, knowledge, dialog):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=180, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output
# Instruction for a chitchat task
instruction = f'Instruction: given a dialog context, you need to response empathically.'
# Leave the knowldge empty
knowledge = ''
random.seed(13)
global_count = 0
MOD = 4
def _norm(x):
    return ' '.join(x.strip().split())


strategies = json.load(open('./strategy.json'))
strategies = [e[1:-1] for e in strategies]
strat2id = {strat: i for i, strat in enumerate(strategies)}
original = json.load(open('./ESConv.json'))

def process_data(d):
    global global_count
    emotion = d['emotion_type']
    problem = d["problem_type"]
    situation = d['situation']
    #init_intensity = int(d['score']['speaker']['begin_intensity'])
    #final_intensity = int(d['score']['speaker']['end_intensity'])

    d = d['dialog']
    dial = []
    history = []
    for uttr in d:
        text = _norm(uttr['content'])
        history.append(text)
        role = uttr['speaker']
        if role == 'seeker':
            dial.append({
                'text': text,
                'speaker': 'usr',
            })
        else:
            dial.append({
                'text': text,
                'speaker': 'sys',
                'strategy': uttr['annotation']['strategy'],
            })
            if '?' == text[-1]:
                if 0 == global_count:
                    print(text)
                    print('generation: ')
                    response = generate(instruction, knowledge, history)
                    print(response)
                global_count = (global_count + 1) % MOD


    res = {
        'emotion_type': emotion,
        'problem_type': problem,
        'situation': situation,
        #'init_intensity': init_intensity,
        #'final_intensity': final_intensity,
        'dialog': dial,
    }
    return res

data = []

# with mp.Pool(processes=mp.cpu_count()) as pool:
for e in tqdm.tqdm(original):2
    data.append(process_data(e))
# with mp.Pool(processes=1) as pool:
#     for e in pool.imap(process_data, tqdm.tqdm(original, total=len(original))):
#         data.append(e)

emotions = Counter([e['emotion_type'] for e in data])
problems = Counter([e['problem_type'] for e in data])
print('emotion', emotions)
print('problem', problems)


random.shuffle(data)
dev_size = int(0.15 * len(data))
test_size = int(0.15 * len(data))
valid = data[:dev_size]
test = data[dev_size: dev_size + test_size]
train = data[dev_size + test_size:]
print("There are ", global_count, "Quesiont markers")
print('train', len(train))

# # {"emotion_type": "anxiety", "problem_type": "job crisis", "situation": "I am on short term disability and I am afraid I will lose my job if I don't go back soon.", "dialog": [{"text": "Hello good afternoon.", "speaker": "usr"}, {"text": "Hi, good afternoon.", "speaker": "sys", "strategy": "Question"}, {"text": "I'm feeling anxious that I am going to lose my job.", "speaker": "usr"}, {"text": "Losing a job is always anxious.", "speaker": "sys", "strategy": "Reflection of feelings"}
# with open('./train_neg.txt', 'w') as f:
#     for e in train:
#         f.write(json.dumps(e) + '\n')
# with open('./sample.json', 'w') as f:
#     json.dump(train[:10], f, ensure_ascii=False, indent=2)

# print('valid', len(valid))
# with open('./valid_neg.txt', 'w') as f:
#     for e in valid:
#         f.write(json.dumps(e) + '\n')

# print('test', len(test))
# with open('./test_neg.txt', 'w') as f:
#     for e in test:
#         f.write(json.dumps(e) + '\n')
