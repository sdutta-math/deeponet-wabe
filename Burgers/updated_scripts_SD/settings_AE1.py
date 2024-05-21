### General
train_epochs = 100000
optuna_epochs = 2000
optuna_trials = 25 #200
optuna_timeout = 1000000000
don_study_name = 'Burgers_DON_optim'  #'DON_optim'
mdon_study_name = 'Burgers_MDON_optim'
ldon_study_name = 'Burgers_LDON_optim'
ae_study_name = 'Burgers_AE_optim1'
train_model = False

###AE & LDON
ae_epochs = 100000
# ae_tuner_epochs = 2000
# ae_trials = 25 #200
ldon_epochs = 1000000
# ldon_tuner_epochs = 2000
# ldon_trials = 200
x_extent_train_LDON = 0.8
t_extent_train_LDON = 1.6
x_extent_val_LDON = 0.8
t_extent_val_LDON = 2
ae_train = True
ldon_train = False
ae_optuna = False
ldon_optuna = False

###GENERAL
loss='mse'
optimizer_str='adam'
scaling=True
scaler_min=0
scaler_max=1
re_train_list=[15,25,35,50,70,100,200,400,800]
re_val_list=[10,1000]
re_test_list=[10,30,75,150,300,600,1000]
x_extent_train=0.8
t_extent_train=1.6
x_extent_val=1
t_extent_val=2
percent_branch=0.05
percent_trunk=0.05

vxn = 300
vtn = 500

percent_branch_test = 0.05
percent_trunk_test = 1


###AUTOENCODER
ae_steps = 5000 #1000
ae_factor = 0.9
ae_learning_rate_decay = True
ae_batch_size = 64
ae_init_lr = 1e-4
ae_number_layers = 2
ae_latent_dim = 36

enc_act = "tanh"
dec_act = "tanh"

ae_optimizer = "Adam"


###CALLBACKS
reduce_patience = 5000

# early_patience = 100000000

model_suffix = 'optuna_1'
