#!/usr/bin/env python
# coding: utf-8


## Load module
import json
import os
import sys
import argparse
import time
from datetime import datetime

import tensorflow as tf
print("TF Version: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.keras.backend.set_floatx('float32')
### -----------------
### SD added
## Used to suppress TF warnings about 'weighted_metrics' and 'sample_weights'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
## USE the above to suppress all warning except ERROR. Do not use if debugging or prototyping
### -----------------

import optuna
from optuna.storages import RDBStorage

import optuna_distributed

from tensorflow.keras.backend import clear_session

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, ScalarFormatter, FormatStrFormatter

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error as mae


# from matplotlib import animation
# matplotlib.rc('animation', html='html5')
# from IPython.display import display
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from tqdm import tqdm


# Plot parameters
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 20,
                     'lines.linewidth': 2,
                     'lines.markersize':10,
                     'axes.labelsize': 16, # fontsize for x and y labels (was 10)
                     'axes.titlesize': 20,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16,
                     'legend.fontsize': 16,
                     'axes.linewidth': 2})

import itertools
colors = itertools.cycle(['r','g','b','m','y','c'])
markers = itertools.cycle(['p','d','o','^','s','x',]) #'D','H','v','*'])

from pathlib import Path, PurePath
try:
    base_dir.exists()
except NameError:
    curr_dir = Path().resolve()
    #base_dir = curr_dir.parent.parent  

if str(curr_dir) == '/p/home/sdutta':
    base_dir = Path("/p/home/sdutta/codes/deeponet-wabe")
else:
    base_dir = curr_dir.parent.parent

scripts_dir = base_dir / "scripts"
work_dir = base_dir / "Burgers" / "updated_scripts_SD"
data_dir = base_dir / "Burgers" / "functions"
model_dir = Path("/p/work1/sdutta") / "jobs"/ "Burgers" / "Saved_DON_models"

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

sys.path.append(str(scripts_dir.absolute()))
sys.path.append(str(work_dir.absolute()))
sys.path.append(str(data_dir.absolute()))


import modified_ldon_mixed as don
import burgers_exact as bg
import data_utils as du
import autoencoder_mixed as ae
import plotting as pt

from importlib import reload as reload

import settings_optuna_PRC as sett
# import settings_AE1 as sett

# Optuna study details
ae_study_name = sett.ae_study_name ## Needed to save/resume Study with RDB backend
ldon_study_name = sett.ldon_study_name ## Needed to save/resume Study with RDB backend
optuna_timeout = sett.optuna_timeout

# Epochs and trials
epochs_ae = sett.ae_epochs
epochs_don = sett.ldon_epochs
if sett.ae_optuna:
    ae_tuner_epochs = sett.ae_tuner_epochs
    ae_trials = sett.ae_trials
if sett.ldon_optuna:
    don_tuner_epochs = sett.ldon_tuner_epochs
    don_trials = sett.ldon_trials

if sett.ae_train or sett.ldon_train:
    model_suffix = sett.model_suffix
     
scaling = sett.scaling
scaler_min = sett.scaler_min
scaler_max = sett.scaler_max
re_train_list = sett.re_train_list 
re_val_list = sett.re_val_list
re_test_list = sett.re_test_list 
x_extent_train = sett.x_extent_train_LDON
t_extent_train = sett.t_extent_train_LDON
x_extent_val = sett.x_extent_val_LDON
t_extent_val = sett.t_extent_val_LDON
percent_branch = sett.percent_branch 
percent_trunk = sett.percent_trunk

vxn = sett.vxn
vtn = sett.vtn

def define_grid(L, T, vxn, vtn):
    vx = np.linspace(0,L,vxn)
    vt = np.linspace(0,T,vtn)
    return vx, vt


def multiple_ae_burgers(Re_list,vxn,VX,vtn,VT):
    
    number_cases = len(Re_list)
    
    burgers_array = np.zeros((number_cases*vtn,vxn))

    id0 = 0
    id1 = vtn
    for Re in Re_list:
        solution = bg.true_solution(VX,VT,Re)
        
        burgers_array[id0:id1,:] = solution
        id0 = id1; id1 = id1 + vtn  
            
    return burgers_array


## Load training data for AE
Re_train = re_train_list
L_train = x_extent_train
T_train = t_extent_train
vx, vt = define_grid(L_train, T_train, vxn, vtn)
VX_train, VT_train = np.meshgrid(vx,vt)
train_data = multiple_ae_burgers(Re_train,vxn,VX_train,vtn,VT_train)


## Load validation data for AE
Re_val = re_val_list
L_val = x_extent_val
T_val = t_extent_val
vx, vt = define_grid(L_val, T_val, vxn, vtn)
VX_val, VT_val = np.meshgrid(vx,vt)
val_data = multiple_ae_burgers(Re_val,vxn,VX_val,vtn,VT_val)

## Load test data for AE
Re_test =  re_test_list
L_test = L_val
T_test = T_val
vx, vt = define_grid(L_test, T_test, vxn, vtn)
VX_test, VT_test = np.meshgrid(vx,vt)
test_data = multiple_ae_burgers(Re_test,vxn,VX_test,vtn,VT_test)


# reload(du)
if scaling:
    scale_min, scale_max = 0, 1
    scaler = du.data_scaler(v_min=scale_min, v_max=scale_max)
    
    ## Flatten and scale
    train_data_scaled, = scaler((train_data,))
    val_data_scaled, test_data_scaled = scaler((val_data, test_data))

    data_min, data_max = scaler.u_min, scaler.u_max
    
    train_data_unscaled, val_data_unscaled, test_data_unscaled = scaler.scale_inverse((train_data_scaled, val_data_scaled, test_data_scaled))
else:
    train_data_scaled, val_data_scaled, test_data_scaled = train_data, val_data, test_data
    scale_min, scale_max = None, None


print(f"Training dataset statistics=====\n")
print(f"Max: {train_data.max()}, Min: {train_data.min()}" )
if scaling:
    print(f"Scaled Max: {train_data_scaled.max()}, Scaled Min: {train_data_scaled.min()}" )
    print(f"Reconstructed Max: {train_data_unscaled.max()}, Reconstructed Min: {train_data_unscaled.min()}" )

print(f"\n\nTest dataset statistics=====\n")
print(f"Max: {test_data.max()}, Min: {test_data.min()}" )
if scaling:
    print(f"Scaled Max: {test_data_scaled.max()}, Scaled Min: {test_data_scaled.min()}" )
    print(f"Reconstructed Max: {test_data_unscaled.max()}, Reconstructed Min: {test_data_unscaled.min()}" )


## Train the AE 

Nn = train_data_scaled.shape[1]
# Nt = train_data_scaled.shape[0]

Nt_train = train_data_scaled.shape[0]
Nt_val = val_data_scaled.shape[0]
Nt_test = test_data_scaled.shape[0]

print(f"Full order dimension: {Nn}")


if (not sett.ae_train) and (not sett.ae_optuna):
    ## Load existing models (if available)
    ae_model_path = os.path.join(model_dir, 'Burgers_AE_2024-05-21_110720_optuna_1')
    
    ae_model,_ = ae.load_model(ae_model_path)
    ae_results = pd.read_csv(ae_model_path+f'/optuna_1_ae_model_history.csv')


from tensorflow.keras.callbacks import Callback

class LRRecorder(Callback):
    """Record current learning rate. """

    def __init__(self):
        self.lr_decay = []

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer._learning_rate(epoch)
        self.lr_decay.append(lr)
        print("The current learning rate is {}".format(lr.numpy()))
    
    def get_lr(self):
        return self.lr_decay


def NN_AE(trial):
    # Define search space
    verbosity_mode = 1

    number_layers = trial.suggest_int("n_layers", 
                                     sett.ae_number_layers_lower,
                                     sett.ae_number_layers_upper)
    latent_dim = trial.suggest_int("latent_space",
                                  sett.latent_dim_lower,
                                  sett.latent_dim_upper,
                                  step = sett.latent_dim_step)
    init_lr = trial.suggest_categorical("ilr", sett.ae_init_lr)
    enc_act = trial.suggest_categorical("enc_activation", sett.enc_act)
    dec_act = trial.suggest_categorical("dec_activation", sett.dec_act)

    set_opt = ae.Optimizer(lr=init_lr)
    optimizerr = "Adam"

    size = np.zeros(number_layers, dtype=int)
    for i in range(number_layers):
        if i==0:
            size[i] = int(Nn)
        else:
            size[i] = int(size[i-1]/2)
    

    model = ae.Autoencoder(latent_dim, enc_act, dec_act, size, )

    model.compile(optimizer = set_opt.get_opt(opt=optimizerr), 
              loss_fn = tf.keras.losses.MeanSquaredError(), #ae.MyNMSELoss(), #
              # metrics=additional_metrics)
             )
    
    return model


# Wrap training step for search
def objective_ae(trial):

    # Clear clutter from previous Keras session graphs.
    clear_session()

    # Build model and optimizer.
    model = NN_AE(trial)

# Add trial for scaling?
    batch_size = trial.suggest_categorical("batch_size",sett.ae_batch_size)
    size_buffer = Nt_train 
    
    train_ds, val_ds = ae.gen_batch_ae(train_data_scaled, val_data_scaled,  
                                     batch_size=batch_size, shuffle_buffer=size_buffer)
    
    history = model.fit(train_ds,
                        validation_data = (val_ds,),
                        batch_size = batch_size, epochs = ae_tuner_epochs) 
    
    score = model.evaluate(val_ds, verbose=0)

    print(score)

    return score

if sett.ae_optuna:
    print("\n\n***** Beginning Optuna study for AE model ******\n")
    # Define search parameters
    # Create a persistent study. An SQLite file `study_name.db' is automatically initialized with a new study record
    #storage = RDBStorage(f"sqlite:///{study_name}:memory:")

    #study = optuna_distributed.from_study(
    ae_study = optuna.create_study(study_name=ae_study_name, direction="minimize",
                                #storage=storage,
                                storage=f'sqlite:///{ae_study_name}.db', 
                                #storage=f'mysql://sdutta@127.0.0.1/{study_name}',
                                load_if_exists=True)

    ae_study.optimize(objective_ae, n_trials = ae_trials, timeout = optuna_timeout, 
                      gc_after_trial=True)

    #
    original_stdout = sys.stdout

    sys.stdout = open("burgers1d_ae_optuna.txt", "w")
    #

    # Print results
    print("Number of finished trials: {}".format(len(ae_study.trials)))

    print("Best trial:")
    trial = ae_study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for keyy, value in trial.params.items():
        print("    {}: {}".format(keyy, value))

    print("Importance:")
    print(optuna.importance.get_param_importances(ae_study))


    #
    sys.stdout = original_stdout # Reset the standard output to its original value
    #

    #export DataFrame to text file (keep header row and index column)
    with open('burgers1d_ae_trials.txt', 'a') as f:
        df_string = ae_study.trials_dataframe().sort_values("value").to_string()
        f.write(df_string)
        

    print(f"Full order dimension: {Nn}")
    print(f"Latent dimension: {trial.params['latent_space']}")
    print("\n------ Optuna study for AE model completed -----\n")

if sett.ae_train:
    print("\n\n***** Beginning to train AE model ******\n")
    steps = sett.ae_steps
    factor = sett.ae_factor
    learning_rate_decay = sett.ae_learning_rate_decay
    batch_size = sett.ae_batch_size
    init_lr = sett.ae_init_lr
    number_layers = sett.ae_number_layers
    latent_dim = sett.ae_latent_dim   ##change to 36

    enc_act = sett.enc_act
    dec_act = sett.dec_act

    if learning_rate_decay == True:
        init_learn_rate = tf.keras.optimizers.schedules.ExponentialDecay(init_lr, 
                                    decay_steps = steps, decay_rate = factor, staircase = True)
    else:
        init_learn_rate = init_lr

    set_opt = ae.Optimizer(lr=init_learn_rate)
    optimizerr = sett.ae_optimizer

    size = np.zeros(number_layers,dtype=int)
    for i in range(number_layers):
        if i==0:
            size[i] = int(Nn)
        else:
            size[i] = int(size[i-1]/2)


    ## Define minibatch generators for training and
    ## validation using Tensorflow Dataset API
    size_buffer = train_data_scaled.shape[0] 
        
    train_ds, val_ds = ae.gen_batch_ae(train_data_scaled, val_data_scaled,  
                                     batch_size = batch_size, shuffle_buffer = size_buffer)

    ae_model = ae.Autoencoder(latent_dim, enc_act, dec_act, size, )

    ae_model.compile(optimizer = set_opt.get_opt(opt=optimizerr), 
                  loss_fn = tf.keras.losses.MeanSquaredError(), #ae.MyNMSELoss(), #
                  # metrics=additional_metrics)
                 )

    save_logs = False
    save_model = False
    
    lr_callback = LRRecorder()

    model_dir_train = model_dir if save_model else None
    log_dir_train = log_dir if save_logs else None

    init_time = time.time()

    history = ae_model.fit(train_ds, #train_data_scaled,
                        validation_data = (val_ds,),
                        epochs = epochs_ae, 
                        callbacks=[lr_callback], 
                        verbose = 1,)
    end_time = time.time()

    ## Visualize AE model results and save Model

    train_time = end_time - init_time
    hrs = int(train_time//3600); rem_time = train_time - hrs*3600
    mins = int(rem_time//60); secs = int(rem_time%60)
    print('Training time: %d H %d M, %d S'%(hrs,mins,secs))

    
    ae_model.build(train_data.shape)

ae_model.summary()


encoded = ae_model.encoder(train_data_scaled).numpy()
decoded = ae_model.decoder(encoded).numpy()

print('\n*********AE inverse decoder reconstruction error*********\n')
print('u  Reconstruction MSE: ' + str(np.mean(np.square(scaler.scale_inverse((decoded,))))))
print('\n')


if sett.ae_train:    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = history.epoch

    # lr = lr_callback.get_lr()
    lr = [history.model.optimizer._learning_rate(ix).numpy() for ix in epochs]  ### Seems to work for ExponentialDecay scheduler
    # lr = [history.model.optimizer.learning_rate(ix).numpy() for ix in epochs] 
else:
    train_loss = ae_results['loss'].to_numpy()
    val_loss = ae_results['valloss'].to_numpy()
    epochs = ae_results['epochs'].to_numpy()
    lr = ae_results['lr'].to_numpy()

if sett.ae_train:
    ## Save the trained AE model
    reload(ae)
    save_model = True
    if save_model:
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = os.path.join(model_dir, "Burgers_AE_"+timestamp+'_'+model_suffix)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        msg = f'Train_list = {re_train_list}, Val_list = {re_val_list}, Test_list = {re_test_list}'\
            +'\nTrains for %dh %dm %ds,'%(hrs,mins,secs)\
            +'\nStep decay LR scheduler starting from %.2e, Batch Size = %d,'%(init_lr, batch_size)\
            +'\nDecay factor = %.3f every %d epochs.'%(factor,steps)+' Trained for %d epochs,'%(len(epochs))\
            +'Scaling to [%d,%d],'%(scale_min, scale_max)\
            +'\nEncoder input is not augmented by parameter value'
        print("\n===========")
        print(msg)
        
        
        model_results = {'loss': train_loss, 'valloss': val_loss,
                         'epochs': epochs, 'msg': msg, 'lr': lr,
                         'umax': scale_max, 'umin': scale_min,
                         'savedir': str(out_dir), 'timestamp': timestamp }

        ae.save_model(ae_model, train_data_scaled.shape, model_results)

        # Creating a DataFrame
        ae_df = pd.DataFrame(model_results)

        # Saving to CSV
        csv_filename = out_dir / Path(model_suffix + '_ae_model_history.csv')
        ae_df.to_csv(csv_filename, index=False)

    print("\n------ AE model training completed -----\n")

if (not sett.ae_train) and (not sett.ae_optuna):
    ## Evaluate loaded AE Model
    encoded = ae_model.encoder(train_data_scaled).numpy()
    decoded = ae_model.decoder(encoded).numpy()

    print('\n*********AE inverse decoder reconstruction error*********\n')
    print('u  Reconstruction MSE: ' + str(np.mean(np.square(scaler.scale_inverse((decoded,))))))
    print('\n')



def multiple_ldon_burgers(Re_list,vxn,VX,L,vtn,VT,T,latent_data):
    
    number_cases = len(Re_list)
    x_grid, t_grid = define_grid(L,T,vxn,vtn)

    for inx,Re in enumerate(Re_list):
        solution = bg.true_solution(VX,VT,Re)
        latent_target = latent_data[inx,:,:]

        
        u0 = solution[0,:]  ## initial solution to be used as input to branch
        
        for id_t,it in tqdm(enumerate(t_grid)):

            if inx==0 and id_t==0:
                b_input = u0
                t_input = it
                target = latent_target[id_t,:]
                
            else:
                b_input = np.vstack([b_input, u0])
                t_input = np.vstack([t_input, it])
                target = np.vstack([target,latent_target[id_t,:]])  
    
    return b_input, t_input, target


if sett.ldon_optuna or sett.ldon_train:

    ## Load data for LDON training

    train_ls = ae_model.encoder(train_data_scaled)
    val_ls = ae_model.encoder(val_data_scaled)
    test_ls = ae_model.encoder(test_data_scaled)

    Nt_train = train_data_scaled.shape[0]
    Nt_val = val_data_scaled.shape[0]
    Nt_test = test_data_scaled.shape[0]

    latent_dim = train_ls.shape[1]
    # assert latent_dim == trial.params['latent_space'], "Optuna trials not available"

    train_ls = np.reshape(train_ls,(len(re_train_list),int(Nt_train/len(re_train_list)),latent_dim))
    val_ls = np.reshape(val_ls,(len(re_val_list),int(Nt_val/len(re_val_list)),latent_dim))
    test_ls = np.reshape(test_ls,(len(re_test_list),int(Nt_test/len(re_test_list)),latent_dim))



    Re_train = re_train_list
    L_train = x_extent_train
    T_train = t_extent_train
    b_train, t_train, target_train = multiple_ldon_burgers(Re_train,vxn,VX_train,L_train,vtn,VT_train,T_train,train_ls)
    
    Re_val = re_val_list
    L_val = x_extent_val
    T_val = t_extent_val

    b_val, t_val, target_val = multiple_ldon_burgers(Re_val,vxn,VX_val,L_val,vtn,VT_val,T_val,val_ls)



    if scaling is True:
        t_scaler = MinMaxScaler(feature_range=( scaler_min,  scaler_max))
        b_scaler = MinMaxScaler(feature_range=( scaler_min,  scaler_max))
        ls_scaler = MinMaxScaler(feature_range=( scaler_min,  scaler_max))

        # changed global scaling, easier for now - PRC
        t_scaler.fit(np.concatenate([t_train,t_val]))
        b_scaler.fit(np.concatenate([b_train,b_val]))
        ls_scaler.fit(np.concatenate([target_train,target_val]))

        t_train = np.squeeze(t_scaler.transform(t_train))
        b_train = b_scaler.transform(b_train)
        target_train = np.squeeze(ls_scaler.transform(target_train))

        t_val = np.squeeze(t_scaler.transform(t_val))
        b_val = b_scaler.transform(b_val)
        target_val = np.squeeze(ls_scaler.transform(target_val))



def NN_LDON(trial):
    # Define search space
    verbosity_mode = 1

    branch_sensors = b_train.shape[1]
    l_factor = trial.suggest_int("l_factor",
                                sett.l_factor_lower,
                                sett.l_factor_upper,
                                step = sett.l_factor_step) 
    l_factor_encoder = trial.suggest_int("l_encoder_factor",
                                sett.l_factor_encoder_lower,
                                sett.l_factor_encoder_upper,
                                step = sett.l_factor_encoder_step) 
    b_number_layers = trial.suggest_int("b_layers", 
                                        sett.b_number_layers_lower, 
                                        sett.b_number_layers_upper,
                                        step = sett.b_number_layers_step) 
    b_actf = trial.suggest_categorical("b_actf", 
                                    sett.b_actf)  
    b_regularizer = trial.suggest_categorical("b_regularizer", 
                                            sett.b_regularizer)  
    b_initializer = trial.suggest_categorical("b_initializer", 
                                            sett.b_initializer) 
    b_encoder_layers = trial.suggest_int("b_encoderlayers", 
                                        sett.b_number_layers_encoder_lower, 
                                        sett.b_number_layers_encoder_upper,
                                        step = sett.b_number_layers_encoder_step) 
    b_encoder_actf = trial.suggest_categorical("b_encoder_actf", 
                                    sett.b_encoder_actf)  
    b_encoder_regularizer = trial.suggest_categorical("b_encoder_regularizer", 
                                            sett.b_encoder_regularizer)  
    b_encoder_init = trial.suggest_categorical("b_encoder_initializer", 
                                            sett.b_encoder_init) 
    
    t_number_layers = trial.suggest_int("t_layers", 
                                        sett.t_number_layers_lower, 
                                        sett.t_number_layers_upper)     
    t_actf = trial.suggest_categorical("t_actf", 
                                    sett.t_actf)  
    t_regularizer = trial.suggest_categorical("t_regularizer", 
                                            sett.t_regularizer)  
    t_initializer = trial.suggest_categorical("t_initializer", 
                                            sett.t_initializer) 
    
    
    t_encoder_layers = trial.suggest_int("t_encoderlayers", 
                                        sett.t_number_layers_encoder_lower, 
                                        sett.t_number_layers_encoder_upper,
                                        step = sett.t_number_layers_encoder_step) 
    
    t_encoder_actf = trial.suggest_categorical("t_encoder_actf", 
                                    sett.t_encoder_actf)  
    t_encoder_regularizer = trial.suggest_categorical("t_encoder_regularizer", 
                                            sett.t_encoder_regularizer)  
    t_encoder_init = trial.suggest_categorical("t_encoder_initializer", 
                                            sett.t_encoder_init) 

    init_lr = trial.suggest_categorical("ilr", sett.init_lr)

    optimizer = tf.keras.optimizers.Adam(init_lr)  

    loss_obj = tf.keras.losses.MeanSquaredError()
    
    nn = don.don_nn(l_factor, 
                    latent_dim, 
                    branch_sensors,
                    b_number_layers, 
                    l_factor*latent_dim, 
                    b_actf, 
                    b_initializer, 
                    b_regularizer, 
                    b_encoder_layers, 
                    l_factor_encoder*latent_dim, 
                    b_encoder_actf, 
                    b_encoder_init, 
                    b_encoder_regularizer, 
                    1, 
                    t_number_layers, 
                    l_factor*latent_dim, 
                    t_actf, 
                    t_initializer, 
                    t_regularizer, 
                    t_encoder_layers, 
                    l_factor_encoder*latent_dim, 
                    t_encoder_actf, 
                    t_encoder_init, 
                    t_encoder_regularizer
                )

    model = don.don_model(nn)

    optimizer = tf.keras.optimizers.Adam(init_lr)  

    loss_obj = tf.keras.losses.MeanSquaredError()

    model.compile(
        optimizer = optimizer,
        loss_fn = loss_obj)
    
    return model

# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).  


# Wrap training step for search
def objective_ldon(trial):

    # Clear clutter from previous Keras session graphs.
    clear_session()

    # Build model and optimizer.
    model = NN_LDON(trial)

    size_buffer = Nt_train 

# Add trial for scaling?
    batch_size = trial.suggest_categorical("batch_size",sett.batch_size)

    dataset = tf.data.Dataset.from_tensor_slices((b_train,t_train, target_train))
    dataset = dataset.shuffle(buffer_size=int(t_train.shape[0])).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((b_val,t_val, target_val))
    val_dataset = val_dataset.batch(batch_size)

    history = model.fit(dataset,validation_data=val_dataset,
                        epochs = sett.ldon_tuner_epochs)
    
    score = model.evaluate(val_dataset, verbose=0)

    print(score)

    return score


if sett.ldon_optuna:
    print("\n\n***** Beginning Optuna study for LDON model ******\n")
    epochs = don_tuner_epochs

    # Deifne search parameters
    study = optuna.create_study(study_name=ldon_study_name, direction="minimize",
                                storage=f'sqlite:///{ldon_study_name}.db', 
                                #storage=f'mysql://sdutta@127.0.0.1/{study_name}',
                                load_if_exists=True)
    
    study.optimize(objective_ldon, n_trials = don_trials, 
                   timeout=optuna_timeout, gc_after_trial=True)
    #
    original_stdout = sys.stdout

    sys.stdout = open("burgers1d_ldon_optuna_5k.txt", "w")
    #

    # Print results
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for keyy, value in trial.params.items():
        print("    {}: {}".format(keyy, value))

    print("Importance:")
    print(optuna.importance.get_param_importances(study))


    #
    sys.stdout = original_stdout # Reset the standard output to its original value
    #

    #export DataFrame to text file (keep header row and index column)
    with open('burgers1d_ldon_trials_5k.txt', 'a') as f:
        df_string = study.trials_dataframe().sort_values("value").to_string()
        f.write(df_string)
    
    print("\n------ Optuna study for LDON model completed -----\n")

## Train the LDON model

if sett.ldon_train:
    
    print("\n\n***** Beginning LDON model training ******\n")

    branch_sensors = b_train.shape[1]
    l_factor = sett.l_factor
    l_factor_encoder = sett.l_factor_encoder
    b_number_layers = sett.b_number_layers
    b_actf =sett.ldon_b_actf
    b_regularizer =sett.ldon_b_regularizer
    b_initializer = sett.ldon_b_initializer
    b_encoder_layers =  sett.b_encoder_number_layers
    b_encoder_actf =  sett.ldon_b_encoder_actf
    b_encoder_regularizer = sett.ldon_b_encoder_regularizer
    b_encoder_initializer = sett.ldon_b_encoder_initializer
    t_number_layers =  sett.t_number_layers 
    t_actf = sett.ldon_t_actf
    t_regularizer = sett.ldon_t_regularizer
    t_initializer = sett.ldon_t_initializer
    t_encoder_layers =  sett.t_encoder_number_layers
    t_encoder_actf = sett.ldon_t_encoder_actf
    t_encoder_regularizer = sett.ldon_t_encoder_regularizer
    t_encoder_initializer = sett.ldon_t_encoder_initializer
    
    ldon_init_lr = sett.ldon_init_lr

    nn = don.don_nn(l_factor, 
            latent_dim, 
            branch_sensors,
            b_number_layers, 
            l_factor*latent_dim, 
            b_actf, 
            b_initializer, 
            b_regularizer, 
            b_encoder_layers, 
            l_factor_encoder*latent_dim, 
            b_encoder_actf, 
            b_encoder_initializer, 
            b_encoder_regularizer, 
            1, 
            t_number_layers, 
            l_factor*latent_dim, 
            t_actf, 
            t_initializer, 
            t_regularizer, 
            t_encoder_layers, 
            l_factor_encoder*latent_dim, 
            t_encoder_actf, 
            t_encoder_initializer, 
            t_encoder_regularizer)



    ldon_model = don.don_model(nn)

    optimizer = tf.keras.optimizers.Adam(ldon_init_lr)  

    loss_obj = tf.keras.losses.MeanSquaredError()

    ldon_model.compile(
            optimizer = optimizer,
            loss_fn = loss_obj,
            #     weighted_metrics=[],
            )


    ldon_batch_size = sett.ldon_batch_size

    dataset = tf.data.Dataset.from_tensor_slices((b_train,t_train, target_train))
    dataset = dataset.shuffle(buffer_size=int(t_train.shape[0])).batch(ldon_batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((b_val,t_val, target_val))
    val_dataset = val_dataset.batch(ldon_batch_size)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9,
            patience=sett.ldon_reduce_patience, min_lr=1e-8, min_delta=0, verbose=1)



    # early_stop = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=1e-8,
    #     patience=500,
    #     verbose=1,
    #     restore_best_weights=True
    # )

    i=1

    timestamp_don = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join(model_dir, 'Burgers_LDON_'+timestamp_don+'_'+model_suffix) 
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    init_time = time.time()

    # Train the model on all available devices.
    ldon_model.fit(dataset, validation_data=val_dataset, epochs=epochs_don,
            callbacks=[reduce_lr, ])  #model_check ])  #early_stop,])  # ])  ## Removed by SD


    end_time = time.time()
    train_time = end_time - init_time
    hrs = int(train_time//3600); rem_time = train_time - hrs*3600
    mins = int(rem_time//60); secs = int(rem_time%60)
    print('Training time: %d H %d M, %d S'%(hrs,mins,secs))


    ldon_model.save(out_dir)#+str(i),id_branch)  
    #np.savez('ldon_history_'+model_suffix, history=ldon_model.history.history, allow_pickle=True,)


    train_loss = ldon_model.history.history['loss']
    val_loss = ldon_model.history.history['val_loss']
    lrate = ldon_model.history.history['lr']
    train_epoch = ldon_model.history.epoch
    ## save dataset

    msg = f'Train_list = {re_train_list}, Val_list = {re_val_list}, Test_list = {re_test_list}'\
                +'\nTrains for %dh %dm %ds,'%(hrs,mins,secs)\
                +'\nReduceLRonPlateau scheduler starting from %.2e, Batch Size = %d,'%(ldon_init_lr, ldon_batch_size)\
                +'\nTrained for %d epochs,'%(len(train_epoch))\
                +'\nScaling to [%d,%d],'%(scaler_min, scaler_max)
    print("\n===========")
    print(msg)

    # Creating a dictionary with your data
    data = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lrate': lrate,
            'train_epoch': train_epoch
            }


    # Creating a DataFrame
    df = pd.DataFrame(data)

    # Saving to CSV
    csv_filename = out_dir / Path(model_suffix + '_ldon_model_history.csv')
    df.to_csv(csv_filename, index=False)

    print("\n------ LDON model training completed -----\n")



