[FedCLS, Ours, Ours+]
1. datapoints: 5 for ours+; 100 for ours; 0 for cls [sweep_aug.py]
2. vote_k_list: [0.1,0.2,0.1,0.5] for ours+; -1 for ours&cls [sweep_aug.py]
3. lr: ours+ small (1e-3); others big (1e-5) [cli.py]
4. bitfit_training: True for ours+; False for ours&cls [modeling.py]
5. augment:  'curriculum' for ours+; 'fixed' for ours [modeling.py]
6. method: "fedpet" and "fedclassifier [sweep_aug.py]"