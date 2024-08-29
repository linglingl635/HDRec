
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

#import fire
import json

from tqdm import tqdm

from llama import Llama, Dialog

# --ckpt_dir
# llama - 2 - 7
# b - chat / --tokenizer_path
# tokenizer.model - -max_seq_len
# 512 - -max_batch_size
# 6


# ckpt_dir = 'llama-2-7b-chat/',
# tokenizer_path = 'tokenizer.model',
def generate_(
    data,
    generator,
    max_batch_size,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None):
    # print(max_batch_size)

    for i in tqdm(range(int(len(data)/max_batch_size)+1), mininterval=2, desc='  - (Generating)   ', leave=False):
    # for i in range(int(len_/max_batch_size)+1):

        input_list = []
        for sample in data[i*max_batch_size:(i+1)*max_batch_size]:
            # print(len(data[i*max_batch_size:(i+1)*max_batch_size]))
            input_list.append([
                # {"role": "system", "content": "Use the template of '{user}'s preference: ...; {item}'s attributes: ..."},

                {
                    "role": "system",
                    "content": "Please generate a short review summary in one sentence based on the given information."
                               "The sentence should start with 'Review summary: '. "
                },
                {
                    "role": "user",
                    "content": "A user bought an item and said '{explanation}'. "
                               "'{user_preference}'. '{item_attribution}'. "
                               "'{user_personality}'. '{item_audience}'. "
                               "Please generate a review summary in one sentence. "
                               "The sentence should start with 'Review summary: '. "
                               .format(explanation=sample['explanation'], user_preference=sample['user_preference'], item_attribution=sample['item_attribution'], user_personality=sample['user_personality'], item_audience=sample['item_audience']),
                },
            ])

        if len(input_list) == 0: continue

        dialogs: List[Dialog] = input_list
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for j, (dialog, result) in enumerate(zip(dialogs, results)):
            strs = result['generation']['content']
            try:
                #print(strs)
                index0 = strs.index('Review summary: ')
                index1 = strs.index('.')
                data[i * max_batch_size + j]['review_summary'] = strs[index0+16:index1]
            except:
                print(strs)
                print(data[i * max_batch_size + j]['explanation'])
                data[i * max_batch_size + j]['review_summary'] = data[i * max_batch_size + j]['explanation']

    return data

def main(
    ckpt_dir: str = 'llama/Meta-Llama-3-8B-Instruct/',
    tokenizer_path: str = 'llama/Meta-Llama-3-8B-Instruct/tokenizer.model',
    max_seq_len: int = 768,
    max_batch_size: int = 50):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    f = open('./data/sports/explanation_rationale3.json', 'r')
    data = json.load(f)
    f.close()

    training_data = data['train']
    # print(len(training_data))
    val_data = data['val']
    test_data = data['test']

    # print(generate_(
    #         training_data[:51],
    #         generator,
    #         max_batch_size)
    #       )

    new_data = {
        'train': generate_(
            training_data,
            generator,
            max_batch_size),

        'val': generate_(
            val_data,
            generator,
            max_batch_size),

        'test': generate_(
            test_data,
            generator,
            max_batch_size,),
    }

    # 保存回JSON文件
    with open('./data/sports/explanation_rationale4.json', 'w') as file:
        json.dump(new_data, file, indent=2)
        file.close()

if __name__ == "__main__":
    #fire.Fire(main)
    main()

