import os
import pickle
import json
import os
from collections import defaultdict
import sys
from force_benchmark.model.load import get_model, get_sae
from transformer_lens.hook_points import HookPoint
from functools import partial

from tqdm import tqdm
import math

import torch as t
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
t.set_grad_enabled(False)

import sys
model_name = sys.argv[1]
lang = sys.argv[2]
save = model_name.split("/")[-1]


ids = t.load(f'force_benchmark/data/{save}/id.{lang}.train.{save}')

print("ids:",ids.shape)


model, model_conf = get_model(model_name=model_name, device=device)

max_length = model_conf["max_model_len"] // 4


l = ids.size(0)
l = min(l, 99999744) // max_length * max_length #
print("l:",l)
input_ids = ids[:l].reshape(-1, max_length)
input_ids = input_ids.to(device)

model_over_zero = t.zeros(model_conf["n_layers"], model_conf["d_mlp"], dtype=t.float32).to(device)
model_over_zero_binary = t.zeros(model_conf["n_layers"], model_conf["d_mlp"], dtype=t.int32).to(device)


print("input_ids:",input_ids.shape)
#sys.exit()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, lst.shape[0], n):
        yield lst[i:i + n,:]

def get_activations_for_dataset(n_batch):

    print(f"Language: {lang}")

    total_batches = math.ceil(len(input_ids) / n_batch)

    for batch in tqdm(chunks(input_ids, n_batch), total=total_batches, desc="Extracting activations from batches: "):
        _, cache = model.run_with_cache(
            batch,
            #stop_at_layer=sae.cfg.hook_layer + 1,
        )
        for idx in range(model_conf["n_layers"]):
            model_activation = cache[f"blocks.{idx}.mlp.hook_post"][:, :, :] #.tolist()
            model_over_zero[idx, :] += model_activation.sum(dim=(0, 1))

            model_activation_binary = (model_activation > 0).sum(dim=(0, 1))
            model_over_zero_binary[idx, :] += model_activation_binary


get_activations_for_dataset(n_batch=2)


model_over_zero_output = dict(n=l, over_zero=model_over_zero.to('cpu'))
model_over_zero_binary_output = dict(n=l, over_zero=model_over_zero_binary.to('cpu'))


t.save(model_over_zero_output, f'force_benchmark/activations/{save}/model_activation.{lang}.train.{save}')
t.save(model_over_zero_binary_output, f'force_benchmark/activations/{save}/model_activation_binary.{lang}.train.{save}')









