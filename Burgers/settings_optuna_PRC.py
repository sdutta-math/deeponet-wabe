train_epochs = 1 #100000
optuna_epochs = 100 #3000
optuna_trials = 2 #200
optuna_timeout = 1000000000
study_name = 'DON_optim'

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

###OPTUNA
neurons_layer_lower = 32
neurons_layer_upper = 256
neurons_layer_step = 32

neurons_layer_encoder_lower = 32
neurons_layer_encoder_upper = 128
neurons_layer_encoder_step = 32

b_number_layers_lower = 2
b_number_layers_upper = 5
b_number_layers_step = 1


b_number_layers_encoder_lower = 2
b_number_layers_encoder_upper = 4
b_number_layers_encoder_step = 1

t_number_layers_encoder_lower = 2
t_number_layers_encoder_upper = 4
t_number_layers_encoder_step = 1

l_factor_lower = 2
l_factor_upper =7
l_factor_step = 1

l_factor_encoder_lower = 1
l_factor_encoder_upper = 5
l_factor_encoder_step = 1

b_actf = ["relu", "elu", "tanh", "swish"]

b_regularizer = ["none", "l1", "l2"]

b_initializer = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]

t_number_layers_lower = 1
t_number_layers_upper = 5

t_actf = ["relu", "elu", "tanh", "swish"]

t_regularizer = ["none", "l1", "l2"]

t_initializer = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]



b_encoder_actf = ["relu", "elu", "tanh", "swish"]

b_encoder_regularizer = ["none", "l1", "l2"]

b_encoder_init = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]


t_encoder_actf = ["relu", "elu", "tanh", "swish"]

t_encoder_regularizer = ["none", "l1", "l2"]

t_encoder_init = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]

batch_size = [64,128,256,512,1024]
# batch_size_lower = 32
# batch_size_upper = 512
# batch_size_step = 32

init_lr = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]


###AUTOENCODER
ae_steps = 5000 #1000
ae_factor = 0.9
ae_learning_rate_decay = True

ae_batch_size = [64,128,256,512,1024]

ae_init_lr = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

ae_number_layers_lower = 2
ae_number_layers_upper = 5
ae_number_layers_step = 1

latent_dim_lower = 12 
latent_dim_upper = 36
latent_dim_step = 12


enc_act = ["relu", "elu", "tanh", "swish"]
dec_act = ["relu", "elu", "tanh", "swish"]

ae_optimizer = "Adam"


###CALLBACKS
reduce_patience = 5000

# early_patience = 100000000

