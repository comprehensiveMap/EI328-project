import matplotlib.pyplot as plt
import json
import numpy as np
import os
plt.switch_backend('agg')


seeds = ['114514', '1919810', '54749110', '1926817', '31415926']
dirs = ['ADDA', 'ADDA_voting', 'AFN', 'BSP', 'DANN', 'DaNN_', 'GTA', 'MCD']
interval_dict = {"ADDA": 1, 'ADDA_voting': 1, 'AFN': 3, "BSP": 5, "DANN": 3, 'DaNN_': 3, 'GTA': 3, "MCD": 2}


def main():
	json_file = open("all_model.json", "r")
	all_dict = json.load(json_file)
	for fold in dirs:
		if not os.path.exists(fold):
			os.makedirs(fold)
		interval = interval_dict[fold]
		all_persons_mean = []
		all_persons_std = []
		max_2_list = []
		print(fold+":")
		for person in ['1', '2', '3']:
			key_mean = fold+person+'_mean'
			key_std = fold+person+'_std'
			mean_acc = np.array(all_dict[key_mean])
			std_acc = np.array(all_dict[key_std])
			epochs = list(range(len(mean_acc)//interval))
			print("person"+person+" max_acc: " + str(max(mean_acc)))
			print("person"+person+" std: " + str(std_acc.mean()))
			max_2_list.append(max(mean_acc))
			mean_acc = np.array([mean_acc[idx*interval: (idx+1)*interval].max() for idx in range(len(epochs))])
			std_acc = np.array([std_acc[idx*interval: (idx+1)*interval].mean() for idx in range(len(epochs))])
			all_persons_mean.append(mean_acc)
			all_persons_std.append(std_acc)
			r1 = list(map(lambda x: x[0]-x[1], zip(mean_acc, std_acc)))
			r2 = list(map(lambda x: x[0]+x[1], zip(mean_acc, std_acc)))
			plt.fill_between(epochs, r1, r2, alpha=0.2)
			plt.plot(epochs, mean_acc, label='person'+person)
		all_persons_mean = [accs.mean() for accs in np.array(all_persons_mean).T]
		all_persons_std = [stds.mean() for stds in np.array(all_persons_std).T]
		print("mean: ", max(all_persons_mean))
		print("mean2: ", np.array(max_2_list).mean())
		print("std:", np.array(all_persons_std).mean())
		# plt.fill_between(epochs, r1, r2, alpha=0.2)
		plt.plot(epochs, all_persons_mean, label='average')
		plt.xlabel('Number of epochs(*'+str(interval)+')')
		plt.ylabel('Accuracy')
		plt.grid()
		plt.legend()
		# plt.savefig(fold+'/'+fold+'.png', dpi=250)
		plt.savefig(fold+'.png', dpi=250)
		plt.clf()


if __name__ == '__main__':
	main()
