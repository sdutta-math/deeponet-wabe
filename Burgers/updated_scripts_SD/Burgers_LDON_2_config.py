## Load module
import json
import os
import sys
import argparse
import time
from datetime import datetime

import tensorflow as tf
tf.keras.backend.set_floatx('float32') 

#import optuna
#from tensorflow.keras.backend import clear_session

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, ScalarFormatter, FormatStrFormatter

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error as mae


from matplotlib import animation
matplotlib.rc('animation', html='html5')
from IPython.display import display
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
# import netCDF4 as nc
from tqdm import tqdm

import settings_config_2 as sett

model_suffix= sett.model_suffix

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
    base_dir = curr_dir.parent.parent

scripts_dir = base_dir / "scripts"
work_dir = base_dir / "Burgers" / "updated_scripts_SD"
data_dir = base_dir / "Burgers" / "functions"
model_dir = base_dir / "Burgers" / "Saved_DON_models"

if not os.path.exists(model_dir):
    os.mkdir(model_dir)


sys.path.append(str(scripts_dir.absolute()))
sys.path.append(str(work_dir.absolute()))
sys.path.append(str(data_dir.absolute()))


import modified_ldon_mixed as don
import burgers_exact as bg
import data_utils as du
import autoencoder_mixed as ae


from importlib import reload as reload


### Set up Training parameters
epochs_ae = sett.epochs_ae
epochs_don = sett.epochs_don

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


def define_grid(L, T, vxn, vtn):
    vx = np.linspace(0,L,vxn)
    vt = np.linspace(0,T,vtn)
    return vx, vt

def plot_bounds_1d(p1,p2,p3,L,T, label1=None, vmin1=None, vmax1=None, name=None):
    
    if vmin1 is None:
        vmin1 = np.amin([p1.min(), p2.min()])
        vmax1 = np.amax([p1.max(), p2.max()])

    fig, ((ax1), (ax2), (ax3)) = plt.subplots(1,3, figsize=(20,3))
    pcm1 = ax1.imshow(p1,cmap='jet',origin='lower',
                      vmin=vmin1, vmax=vmax1, extent=(0,T,0,L), aspect = 0.79)
    ax1.yaxis.set_ticks(np.arange(0,L+0.1,1))
    fig.colorbar(pcm1,ax=ax1)
    ax1.set_title('Truth' +'\n'+ '%s = %.2f'%('Re', label1)+'\n'+
                 '%.4f<u<%.4f'%(tf.reduce_min(p1).numpy(), tf.reduce_max(p1).numpy()))
    pcm2 = ax2.imshow(p2,cmap='jet',origin='lower',
                      vmin=vmin1, vmax=vmax1, extent=(0,T,0,L), aspect = 0.79)
    ax2.yaxis.set_ticks(np.arange(0,L+0.1,1))

    fig.colorbar(pcm2,ax=ax2)
    ax2.set_title('Prediction' +'\n'+ '%s = %.2f'%('Re', label1)+'\n'
                 '%.4f<u<%.4f'%(tf.reduce_min(p2).numpy(), tf.reduce_max(p2).numpy()))
    pcm3 = ax3.imshow(p3,cmap='coolwarm',origin='lower',
                      vmin=-0.05, vmax=0.05, extent=(0,T,0,L), aspect = 0.79)
    ax3.yaxis.set_ticks(np.arange(0,L+0.1,1))

    cbar = fig.colorbar(pcm3,ax=ax3)
    ax3.set_title('Relative Error' +'\n'+ '%s = %.2f'%('Re', label1))
    
    ax1.set_ylabel('$x$',fontsize=18)
    ax1.set_xlabel('$t$',fontsize=18)
    ax2.set_xlabel('$t$',fontsize=18) 
    ax3.set_xlabel('$t$',fontsize=18)
    
    fig.tight_layout()
    plt.savefig(name)

def plot_spcaetime_1d(p1,p2,p3,p4,p5,p6,T,L, colormap='jet', label1=None, label2=None, vmin1=None, vmax1=None, name=None):
    """
    Plot space-time 2d plots of 1D solutions
    Row1 : Predicted, True, Error for Soln1
    Row2 : Predicted, True, Error for Soln2
    """
    f = plt.figure(figsize=(15,10))
    gs = gridspec.GridSpec(3, 3, )
    gs.update(wspace=0.2, hspace=0.2) # set the spacing between axes.

    if vmin1 is None:
        vmin1 = np.amin([p1.min(), p2.min(), p3.min(), p4.min()])
        vmax1 = np.amax([p1.max(), p2.max(), p3.max(), p4.max()])

    ax1 = plt.subplot(gs[0, 0]);
    f1= ax1.imshow(p1,cmap=colormap,origin='lower',vmin=vmin1, vmax=vmax1, extent=(0,T,0,L), aspect = 0.59)
    ax1.yaxis.set_ticks(np.arange(0,L+0.1,1)); ax1.xaxis.set_ticks(np.arange(0,T+0.1,1))
    ax1.set_title('%s=%.2f'%('Re',label1[0]))

    ax2 = plt.subplot(gs[0, 1]);
    f2 = ax2.imshow(p2,cmap=colormap,origin='lower',vmin=vmin1, vmax=vmax1, extent=(0,T,0,L), aspect = 0.59)
    ax2.yaxis.set_ticks(np.arange(0,L+0.1,1)); ax2.xaxis.set_ticks(np.arange(0,T+0.1,1))
    ax2.set_title('%s=%.2f'%('Re',label1[1]))

    ax3 = plt.subplot(gs[1, 0]);
    f3= ax3.imshow(p3,cmap=colormap,origin='lower',vmin=vmin1, vmax=vmax1, extent=(0,T,0,L), aspect = 0.59)
    ax3.set_xticklabels([]); ax3.set_yticklabels([])
#     ax3.yaxis.set_ticks(np.arange(0,1.1,1))
    ax3.set_title('%s=%.2f'%('Re',label1[2]))

    ax4 = plt.subplot(gs[1, 1]);
    f4 = ax4.imshow(p4,cmap=colormap,origin='lower',vmin=vmin1, vmax=vmax1, extent=(0,T,0,L), aspect = 0.59)
    ax4.set_xticklabels([]); ax4.set_yticklabels([])
    ax4.set_title('%s=%.2f'%('Re',label1[3]))
    
    ax5 = plt.subplot(gs[2, 0]);
    f5= ax5.imshow(p5,cmap=colormap,origin='lower',vmin=vmin1, vmax=vmax1, extent=(0,T,0,L), aspect = 0.59)
    ax5.set_xticklabels([]); ax5.set_yticklabels([])
#     ax3.yaxis.set_ticks(np.arange(0,1.1,1))
    ax5.set_title('%s=%.2f'%('Re',label1[4]))

    ax6 = plt.subplot(gs[2, 1]);
    f6 = ax6.imshow(p6,cmap=colormap,origin='lower',vmin=vmin1, vmax=vmax1, extent=(0,T,0,L), aspect = 0.59)
    cbar1 = f.colorbar(f6, ax=list((ax1, ax2, ax3, ax4, ax5, ax6)),orientation='horizontal',aspect=50, pad=0.1)
    ax6.set_xticklabels([]); ax6.set_yticklabels([])
    ax6.set_title('%s=%.2f'%('Re',label1[5]))

    ax1.set_ylabel('$x$',fontsize=18); #ax2.set_ylabel('$x$',fontsize=18);
    ax3.set_ylabel('$x$',fontsize=18); #ax4.set_ylabel('$x$',fontsize=18);

    ax5.set_xlabel('$t$',fontsize=18); ax6.set_xlabel('$t$',fontsize=18); 
    ax5.set_ylabel('$x$',fontsize=18); #ax6.set_ylabel('$x$',fontsize=18);
    plt.savefig(name)



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
vxn = sett.vxn
vtn = sett.vtn
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



reload(du)
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
Nt = train_data_scaled.shape[0]

print(f"Full order dimension: {Nn}")



if sett.ae_train:

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

    model_dir_train = model_dir if save_model else None
    log_dir_train = log_dir if save_logs else None

    init_time = time.time()

    history = ae_model.fit(train_ds, #train_data_scaled,
                        validation_data = (val_ds,), #(val_data_scaled, val_data_scaled),
                        epochs = epochs_ae, #callbacks=[es], 
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

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = history.epoch
    # lr = [history.model.optimizer._learning_rate(ix).numpy() for ix in epochs]  ### Seems to work for ExponentialDecay scheduler
    lr = [history.model.optimizer.learning_rate(ix).numpy() for ix in epochs] 
    # lr = history.history['lr']


    ## Save the trained AE model
    reload(ae)
    save_model = True
    if save_model:
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = os.path.join(model_dir, "Burgers_AE_"+timestamp+model_suffix)
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
        csv_filename = model_suffix + '_ae_model_history.csv'
        ae_df.to_csv(csv_filename, index=False)





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


if sett.ldon_train:

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

    ## Train the LDON model


    branch_sensors = b_train.shape[1]
    l_factor = sett.l_factor
    b_number_layers = sett.b_number_layers
    b_actf =sett.b_actf
    b_regularizer =sett.b_regularizer
    b_initializer = sett.b_initializer
    b_encoder_layers =  sett.b_encoder_number_layers
    b_encoder_actf =  sett.b_encoder_actf
    b_encoder_regularizer = sett.b_encoder_regularizer
    b_encoder_initializer = sett.b_encoder_initializer
    t_number_layers =  sett.t_number_layers 
    t_actf = sett.t_actf
    t_regularizer = sett.t_regularizer
    t_initializer = sett.t_initializer
    t_encoder_layers =  sett.t_encoder_number_layers
    t_encoder_actf = sett.t_encoder_actf
    t_encoder_initializer = sett.t_encoder_initializer
    t_encoder_regularizer = sett.t_encoder_regularizer
    init_lr = sett.init_lr

    nn = don.don_nn(l_factor, 
                    latent_dim, 
                    branch_sensors,
                    b_number_layers, 
                    l_factor*latent_dim, 
                    b_actf, 
                    b_initializer, 
                    b_regularizer, 
                    b_encoder_layers, 
                    l_factor*latent_dim, 
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
                    l_factor*latent_dim, 
                    t_encoder_actf, 
                    t_encoder_initializer, 
                    t_encoder_regularizer)



    ldon_model = don.don_model(nn)

    optimizer = tf.keras.optimizers.Adam(init_lr)  

    loss_obj = tf.keras.losses.MeanSquaredError()

    ldon_model.compile(
        optimizer = optimizer,
        loss_fn = loss_obj,
    #     weighted_metrics=[],
    )
        

    batch_size = sett.batch_size

    dataset = tf.data.Dataset.from_tensor_slices((b_train,t_train, target_train))
    dataset = dataset.shuffle(buffer_size=int(t_train.shape[0])).batch(batch_size)
       
    val_dataset = tf.data.Dataset.from_tensor_slices((b_val,t_val, target_val))
    val_dataset = val_dataset.batch(batch_size)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                        patience=sett.reduce_patience, min_lr=1e-8, min_delta=0, verbose=1)



    # early_stop = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=1e-8,
    #     patience=500,
    #     verbose=1,
    #     restore_best_weights=True
    # )

    i=1

    timestamp_don = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join(model_dir, 'Burgers_ldon'+timestamp_don+model_suffix) 
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


    # model.save(out_dir,id_branch,)
    ldon_model.save(out_dir)#+str(i),id_branch)  
    np.savez('ldon_history_', history=ldon_model.history.history, allow_pickle=True,)


    train_loss = ldon_model.history.history['loss']
    val_loss = ldon_model.history.history['val_loss']
    lrate = ldon_model.history.history['lr']
    train_epoch = ldon_model.history.epoch
    ## save dataset

    # Creating a dictionary with your data
    data = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'lrate': lrate,
        'train_epoch': train_epoch
    }

    import pandas as pd

    # Creating a DataFrame
    df = pd.DataFrame(data)

    # Saving to CSV
    csv_filename = model_suffix + 'ldon_model_history.csv'
    df.to_csv(csv_filename, index=False)



