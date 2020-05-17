import os
import json
import numpy as np

seeds = ['114514', '1919810', '54749110', '1926817', '31415926']
# dirs = ['ADDA', 'BSP', 'DANN', 'MCD', 'DaNN_']
dirs = ['ADDA', 'ADDA_voting', 'AFN', 'BSP', 'DANN', 'DaNN_', 'GTA', 'MCD']

json_dict = dict()

for fold in dirs:
    for person in ['1', '2', '3']:
        accs_list = []
        for seed in seeds:
            file_path = fold+'/'+seed+'/'+'acc'+person+'.json'
            with open(file_path, "r") as f:
                jd = json.load(f)
                accs_list.append(jd['test_acc'])
        # accs_list = np.array(accs_list).reshape(-1, len(seeds))
        accs_list = np.array(accs_list).T
        mean_accs = np.array([accs.mean() for accs in accs_list])
        std_accs = np.array([accs.std() for accs in accs_list])
        json_dict[fold+person+"_mean"] = list(mean_accs)
        json_dict[fold+person+"_std"] = list(std_accs)

with open("all_model.json", "w") as f:
    json.dump(json_dict, f)