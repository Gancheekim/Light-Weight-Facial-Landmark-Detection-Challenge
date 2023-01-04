

cfg = {
    # set a random seed to get a fixed initialization 
    'seed': 30678,
    
    # training hyperparameters
    'batch_size': 38,
    'lr': 1e-3,
    'milestones': [5, 10, 15],
    'milestones_gamma': 0.45,
    'num_epoch': 40,   
    'save_path': "./bestmodel"
}