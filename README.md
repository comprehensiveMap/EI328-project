# EI328-project
================= Our experiments are conducted using Pytorch 1.4.0====================\\
To run our experiment over all advanced models, please first activate your pytorch evirenment, and just use:\\
(torch)$ python3 train_seeds.py\\
Then you will get accuracy records in model files.\\
Next, use:\\
(torch)$ python3 compute_mu_std.py\\
to do the results processing work.\\
After this, you will see all_model.json on current fold.\\
Finally, use:\\
(torch)$ python3 draw_all.py\\
to do the visualization work and get mean accuracy and standard deviation.\\
(torch)$ python3 t-SNE.py\\
to draw the pictures of case study.\\
If you want to run our baselines, just open the .ipynb files on fold \baselines.
