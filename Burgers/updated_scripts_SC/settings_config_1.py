#DON and MDON
train_epochs = 100000

# LDON
epochs_ae = 100000
epochs_don = 100000

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

#AE
ae_steps = 2500 #1000
ae_factor = 0.9
ae_learning_rate_decay = True
ae_batch_size = 256
ae_init_lr = 1e-5
ae_number_layers = 3
ae_latent_dim = 36   ##change to 36

enc_act = 'relu' 
dec_act = 'relu'

ae_optimizer = "Adam"

###NETWROKS
model_suffix = 'config_1'

neurons_layer= 128

b_number_layers = 5

b_actf = 'relu'

b_regularizer = 'none'

b_initializer = 'glorot_normal'

t_number_layers = 5

t_actf = 'relu'

t_regularizer = 'none'

t_initializer = 'glorot_normal'


b_encoder_number_layers = 3

b_encoder_actf = 'relu'

b_encoder_regularizer = 'none'

b_encoder_initializer = 'glorot_normal'

t_encoder_number_layers = 3

t_encoder_actf = 'relu'

t_encoder_regularizer = 'none'

t_encoder_initializer = 'glorot_normal'


l_factor = 4


init_lr = 1e-4

batch_size = 256

###CALLBACKS
reduce_patience = 5000

early_patience = 100000

