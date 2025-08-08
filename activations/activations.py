import os
import pickle
import json
import os
from collections import defaultdict
import sys
from force_benchmark.model.load import get_model, get_sae

import torch as t
device = "cpu" #t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
t.set_grad_enabled(False)

import sys
model_name = sys.argv[1]
lang = sys.argv[2]
save = model_name.split("/")[-1]
path_to_saes = f"/pfss/mlde/workspaces/mlde_wsp_P_DFKI_Darmstadt/ya98xoke/multi_low_res_cache/{save}/models"


compnent_id = 1
factor = 8





#os.makedirs(results_path, exist_ok=True)
#os.makedirs(os.path.join(results_path,"sae"), exist_ok=True)
#os.makedirs(os.path.join(results_path,"llm"), exist_ok=True)


# with open(dataset_path) as f:
#     dataset = json.load(f)

ids = t.load(f'force_benchmark/data/{save}/id.{lang}.train.{save}')

print("ids:",ids.shape)


model, model_conf = get_model(model_name=model_name, device=device)

max_length = model_conf["max_model_len"]


l = ids.size(0)
l = min(l, 99999744) // max_length * max_length
print("l:",l)
input_ids = ids[:l].reshape(-1, max_length)
input_ids = input_ids.to(device)

model_over_zero = t.zeros(model_conf["n_layers"], model_conf["d_model"], dtype=t.float32).to(device)
sae_over_zero = t.zeros(model_conf["n_layers"], model_conf["d_model"]*factor, dtype=t.float32).to(device)
model_over_zero_binary = t.zeros(model_conf["n_layers"], model_conf["d_model"], dtype=t.int32).to(device)
sae_over_zero_binary = t.zeros(model_conf["n_layers"], model_conf["d_model"]*factor, dtype=t.int32).to(device)

print("input_ids:",input_ids.shape)
#sys.exit()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, lst.shape[0], n):
        yield lst[i:i + n,:]

def get_activations_for_dataset(layer):

    print(f"Layer: {layer}")
    
    #try:
    sae, sae_conf = get_sae(layer=layer,path_to_saes=path_to_saes, factor=factor, compnent_id=compnent_id, device=device)

    # Creating the default dictionary with the default tensor factory
    #results[layer] = defaultdict(default_tensor)
    model_results = defaultdict(list)
    sae_results = defaultdict(list)

    print(f"Language: {lang}")

    for batch in chunks(input_ids, 16): 
        #print(batch.shape)

        _, cache = model.run_with_cache_with_saes(
            batch,
            saes=[sae],
            stop_at_layer=sae.cfg.hook_layer + 1,
        )
        #sys.exit()
        #print(cache.keys())
        idx = layer
        model_activation = cache[f"{sae.cfg.hook_name}.hook_sae_input"][:, :, :] #.tolist()
        model_over_zero[idx, :] += model_activation.sum(dim=(0, 1))
        sae_activation = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"][:, :, :] #.tolist()
        sae_over_zero[idx, :] += sae_activation.sum(dim=(0, 1))

        model_activation_binary = (model_activation > 0).sum(dim=(0, 1))
        model_over_zero_binary[idx, :] += model_activation_binary
        sae_activation_binary = (sae_activation > 0).sum(dim=(0, 1))
        sae_over_zero_binary[idx, :] += sae_activation_binary

    # save_path = os.path.join(results_path,"sae",f"alltokens-F{factor}-Comp{compnent_id}-L{layer}.pkl")
    # print(save_path)
    # with open(save_path,"wb") as f:
    #     pickle.dump(sae_results,f)

    # save_path = os.path.join(results_path,"llm",f"alltokens-F{factor}-Comp{compnent_id}-L{layer}.pkl")
    # print(save_path)
    # with open(save_path,"wb") as f:
    #     pickle.dump(model_results,f)

    #except Exception as e:
    #    print(e)
        

#results = 

for layer in range(model_conf["n_layers"]):
    get_activations_for_dataset(layer)


model_over_zero_output = dict(n=l, over_zero=model_over_zero.to('cpu'))
sae_over_zero_output = dict(n=l, over_zero=sae_over_zero.to('cpu'))
model_activation_binary_output = dict(n=l, over_zero=model_activation_binary.to('cpu'))
sae_over_zero_binary_output = dict(n=l, over_zero=sae_over_zero_binary.to('cpu'))


t.save(model_over_zero_output, f'force_benchmark/activations/{save}/model_activation.{lang}.train.{save}')
t.save(sae_over_zero_output, f'force_benchmark/activations/{save}/sae_activation.{lang}.train.{save}')
t.save(model_activation_binary_output, f'force_benchmark/activations/{save}/mode_activation_binary.{lang}.train.{save}')
t.save(sae_over_zero_binary_output, f'force_benchmark/activations/{save}/sae_activation_binary.{lang}.train.{save}')

# save_path = f"./activations/alltokens-{dataset_path.split('/')[-1].replace('.json','')}-{factor}-{compnent}.pkl"

#with open(save_path,"wb") as f:
#    pickle.dump(results,f)








