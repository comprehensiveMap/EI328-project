import os

seeds = ['114514', '1919810', '54749110', '1926817', '31415926']
dirs = ['ADDA', 'AFN', 'BSP', 'DANN', 'DaNN_', 'GTA', 'MCD', 'ADDA_voting']
# dirs = ['ADDA_voting']


for fold in dirs:
    os.chdir(fold)
    for seed in seeds:
        if not os.path.exists(seed):
            os.makedirs(seed)
        for person in range(1, 4):
            print("Starting training " + fold + " " + seed + " " + str(person))
            os.system("python3 main.py --person " + str(person) + " --seed " + seed)
            # # if you wanna run nohup (parallel), please make sure you run only ONE model at the same time, or the memory of your GPU
            # # will be exploded (if you have a GPU with memory over 10G, then it is OK to train all the models at the same time...) 
            # os.system("nohup python3 -u main.py --person " + str(person) + " --seed " + seed + " >> run.log &")
    os.chdir("..")