train_epochs = 100000
optuna_epochs = 3000
optuna_trials = 200

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

b_number_layers_lower = 1
b_number_layers_upper = 5

b_actf = ["relu", "elu", "tanh", "swish"]

b_regularizer = ["none", "l1", "l2"]

b_initializer = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]

t_number_layers_lower = 1
t_number_layers_upper = 5

t_actf = ["relu", "elu", "tanh", "swish"]

t_regularizer = ["none", "l1", "l2"]

t_initializer = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]

b_encoder_number_layers_lower = 1
b_encoder_number_layers_upper = 5

b_encoder_actf = ["relu", "elu", "tanh", "swish"]

b_encoder_regularizer = ["none", "l1", "l2"]

b_encoder_initializer = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]

t_encoder_number_layers_lower = 1
t_encoder_number_layers_upper = 5

t_encoder_actf = ["relu", "elu", "tanh", "swish"]

t_encoder_regularizer = ["none", "l1", "l2"]

t_encoder_initializer = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]

latent_dim_lower = 12 
latent_dim_upper = 36
latent_dim_step = 12

init_lr = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

batch_size_lower = 32
batch_size_upper = 512
batch_size_step = 32

###CALLBACKS
reduce_patience = 100

early_patience = 100

