optuna_rank = "2_configuration" 

from sklearn.preprocessing import MinMaxScaler

import json
import os
import sys
import argparse
import time
from datetime import datetime

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if tf.__version__ == '2.16.1':
    import tf_keras as keras
else:
    import tensorflow.keras as keras

### -----------------
### SD added
## Used to suppress TF warnings about 'weighted_metrics' and 'sample_weights'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
## USE the above to suppress all warning except ERROR. Do not use if debugging or prototyping
### -----------------

# import keras_tuner
tf.keras.backend.set_floatx('float64') 


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, ScalarFormatter, FormatStrFormatter

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as mae


from matplotlib import animation
matplotlib.rc('animation', html='html5')
from IPython.display import display
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
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
except:
    curr_dir = Path().resolve()
    base_dir = curr_dir.parent.parent  

scripts_dir = base_dir / "scripts"
work_dir = base_dir / "Burgers"
data_dir = base_dir / "Burgers" / "functions"
model_dir = base_dir / "Burgers" / "Saved_DON_models"

if not os.path.exists(model_dir):
    os.mkdir(model_dir)


sys.path.append(str(scripts_dir.absolute()))
sys.path.append(str(work_dir.absolute()))
sys.path.append(str(data_dir.absolute()))


#data_dir = "/work/08372/scai/ls6/deeponet-wabe-main/data/burgers1d"
#fig_dir  = "/work/08372/scai/ls6/deeponet-wabe-main/figures"
#scripts_dir  = "/work/08372/scai/ls6/deeponet-wabe-main/scripts"
#work_dir = "/work/08372/scai/ls6/deeponet-wabe-main/Burgers"
#model_dir = "/work/08372/scai/ls6/deeponet-wabe-main/model"

#if not os.path.exists(model_dir):
#    os.mkdir(model_dir)


# sys.path.append(str(scripts_dir.absolute()))
# sys.path.append(str(work_dir.absolute()))
# sys.path.append(str(data_dir.absolute()))

#sys.path.append(r'/work/08372/scai/ls6/deeponet-wabe-main/scripts')
#sys.path.append(r'/work/08372/scai/ls6/deeponet-wabe-main/Burgers')
# sys.path.append(r'/work/08372/scai/ls6/deeponet-wabe-main/data/burgers1d')
#sys.path.append(r'/work/08372/scai/ls6/deeponet-wabe-main/Burgers/functions')

import modified_don as don
import burgers_exact as bg
#import settings as sett

case='Train' #or 'Predict'

if case == 'Predict':
    
    ## Replace below with appropriate timestamp of saved model
    model_path = model_dir+'_burgers1d_'+timestamp_don+optuna_rank

    loaded_model = tf.keras.models.load_model(model_path)
    branch_id = np.load(model_path+'/branch_id.npy')

train_epochs = sett.train_epochs
loss = sett.loss                    ## NOT USED
optimizer_str = sett.optimizer_str  ## NOT USED
scaling = sett.scaling
scaler_min = sett.scaler_min
scaler_max = sett.scaler_max
re_train_list = sett.re_train_list
re_val_list = sett.re_val_list
re_test_list = sett.re_test_list
x_extent_train = sett.x_extent_train
t_extent_train = sett.t_extent_train
x_extent_val = sett.x_extent_val
t_extent_val = sett.t_extent_val
percent_branch = sett.percent_branch
percent_trunk = sett.percent_trunk


def define_grid(L, T, vxn, vtn):
    vx = np.linspace(0,L,vxn)
    vt = np.linspace(0,T,vtn)
    return vx, vt

def plot_bounds_1d(p1,p2,p3,L,T, label1=None, vmin1=None, vmax1=None, name=None):
    
    if vmin1 is None:
        vmin1 = np.amin([p1.min(), p2.min()])
        vmax1 = np.amax([p1.max(), p2.max()])

    fig, ((ax1), (ax2), (ax3)) = plt.subplots(1,3, figsize=(20,10))
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
    #plt.savefig(name)

def plot_spcaetime_1d(data_list, T, L, colormap='jet', label=None, vmin=None, vmax=None, name=None):
    """
    Plot space-time 2d plots of 1D solutions
    """
    num_plots = len(data_list)

    # Calculate vmin and vmax if not provided
    if vmin is None:
        vmin = min(np.amin(data.min()) for data in data_list)
    if vmax is None:
        vmax = max(np.amax(data.max()) for data in data_list)

    num_cols = 3  # Number of columns for subplots
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate number of rows needed

    f = plt.figure(figsize=(15, 5 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_cols)
    gs.update(wspace=0.2, hspace=0.2)  # set the spacing between axes.

    axes = []  # Store axes for colorbar

    for i, data in enumerate(data_list):
        ax = plt.subplot(gs[i // num_cols, i % num_cols])
        im = ax.imshow(
            data,
            cmap=colormap,
            origin='lower',
            vmin=vmin,
            vmax=vmax,
            extent=(0, T, 0, L),
            aspect=0.59
        )
        ax.yaxis.set_ticks(np.arange(0, L + 0.1, 1))
        ax.xaxis.set_ticks(np.arange(0, T + 0.1, 1))
        ax.set_title('%s=%.2f' % ('Re', label[i]) if label else f'Plot {i+1}')

        if i // num_cols == num_rows - 1:  # Set x labels for bottom row
            ax.set_xlabel('$t$', fontsize=18)
        if i % num_cols == 0:  # Set y labels for leftmost column
            ax.set_ylabel('$x$', fontsize=18)

        axes.append(ax)

    # Create colorbar using all axes
    cbar = f.colorbar(im, ax=axes, orientation='horizontal', aspect=50, pad=0.1)
    #plt.savefig(name)

def multiple_burgers(Re_list,vxn,vx,vtn,vt,percent_branch,percent_trunk,id_branch=None):
    
    number_cases = len(Re_list)
    branch_sensors = int(vxn*percent_branch)
    trunk_sensors = int(vtn*vxn*percent_trunk)
    
    burgers_array = np.zeros((number_cases,vxn,vtn))
    burgers_flatten = np.zeros((number_cases*vxn*vtn))
    count = 0 
    id0 = 0
    id1 = vxn*vtn
    
    for Re in Re_list:
        solution = np.zeros((vxn,vtn))
        for ix,vxi in enumerate(vx):
            solution[ix] = bg.true_solution(vxi,vt,Re)
        burgers_array[count] = solution
        burgers_flatten[id0:id1] = (solution.flatten())
        count = count + 1
        id0 = id1
        id1 = id1 + vxn*vtn
        
    b0 = burgers_array[:,:,0] #u0
    
    if id_branch is not None:
        id_b=np.zeros(np.shape(id_branch),dtype=int)
        for i in range(len(id_branch[0])):
            smallest=np.absolute(vx-id_branch[1][i])
            id_b[0][i]=int(np.argmin(smallest))
        id_b[1]=id_branch[1]    
    else:
        id_b = np.sort(np.random.choice(vxn, branch_sensors, replace=False))
        b_coords = vx[id_b]
        id_b = [id_b, b_coords]

    b0_train = b0[:,id_b[0]]

    
    T, X = np.meshgrid(vt,vx)
    for i in range(number_cases):
        b0_vector = np.tile(np.expand_dims(b0_train[i],1),vtn*vxn).T
        
        id_t = np.sort(np.random.choice(vtn*vxn, trunk_sensors, replace=False))
        
        if i == 0:
            b_input = b0_vector[id_t]
            t_input = np.hstack([T.flatten()[id_t,None], X.flatten()[id_t,None]])
            target = burgers_array[i,:,:].flatten()[id_t,None]

        else:
            b_input = np.vstack([b_input,b0_vector[id_t]])
            t_input = np.vstack([t_input,np.hstack([T.flatten()[id_t,None], X.flatten()[id_t,None]])])
            target = np.vstack([target,burgers_array[i,:,:].flatten()[id_t,None]])        
    
    return burgers_array, burgers_flatten, b_input, t_input, target, id_b

# Re_train = [15,25,50,75,100,150,200,300,400,600,800,900]
Re_train =  re_train_list
L_train =  x_extent_train
T_train =  t_extent_train
vxn = 300
vtn = 500
vx, vt = define_grid(L_train, T_train, vxn, vtn)
percent_branch =  percent_branch
percent_trunk =  percent_trunk
burgers_array_train, burgers_flatten_train, b_train, \
                    t_train, target_train, id_branch = multiple_burgers(Re_train,vxn,vx,vtn,vt,
                                                            percent_branch,percent_trunk)

if case == 'Predict':
    id_branch=branch_id
    
Re_val =  re_val_list
L_val =  x_extent_val
T_val =  t_extent_val
vxn = 300
vtn = 500
vx, vt = define_grid(L_val, T_val, vxn, vtn)
percent_branch =  percent_branch
percent_trunk =  percent_trunk
burgers_array_val, burgers_flatten_val, b_val, \
                    t_val, target_val, _ = multiple_burgers(Re_val,vxn,vx,vtn,vt,
                                                         percent_branch,percent_trunk,id_branch=id_branch)

Re_test =  re_test_list
L_test = L_val
T_test = T_val
vxn = 300
vtn = 500
vx, vt = define_grid(L_test, T_test, vxn, vtn)
percent_branch = 0.05
percent_trunk = 1
burgers_array_test, burgers_flatten_test, b_test, \
                    t_test, target_test, _ = multiple_burgers(Re_test,vxn,vx,vtn,vt, 
                                                              percent_branch,percent_trunk,id_branch)

if  scaling is True:
    x_scaler = MinMaxScaler(feature_range=( scaler_min,  scaler_max))
    t_scaler = MinMaxScaler(feature_range=( scaler_min,  scaler_max))
    u_scaler = MinMaxScaler(feature_range=( scaler_min,  scaler_max))
    b_scaler = MinMaxScaler(feature_range=( scaler_min,  scaler_max))

    x_scaler.fit(np.expand_dims(t_val[:,0],1))
    t_scaler.fit(np.expand_dims(t_val[:,1],1))
    u_scaler.fit(target_val)
    b_scaler.fit(b_val)

    t_train[:,0] = np.squeeze(x_scaler.transform(np.expand_dims(t_train[:,0],1)))
    t_train[:,1] = np.squeeze(t_scaler.transform(np.expand_dims(t_train[:,1],1)))
    b_train = b_scaler.transform(b_train)
    target_train = u_scaler.transform(target_train)
    
    t_val[:,0] = np.squeeze(x_scaler.transform(np.expand_dims(t_val[:,0],1)))
    t_val[:,1] = np.squeeze(t_scaler.transform(np.expand_dims(t_val[:,1],1)))
    b_val = b_scaler.transform(b_val)
    target_val = u_scaler.transform(target_val)
    
    t_test[:,0] = np.squeeze(x_scaler.transform(np.expand_dims(t_test[:,0],1)))
    t_test[:,1] = np.squeeze(t_scaler.transform(np.expand_dims(t_test[:,1],1)))
    b_test = b_scaler.transform(b_test)
    target_test = u_scaler.transform(target_test)       

branch_sensors = int(vxn* percent_branch)

nn = don.don_nn(branch_input_shape=branch_sensors,
            branch_output_shape = sett.neurons_layer,
            b_number_layers = sett.b_number_layers,
            b_neurons_layer = sett.neurons_layer,
            b_actf = sett.b_actf,
            b_init = sett.b_initializer,
            b_regularizer = sett.b_regularizer,

            b_encoder_layers= sett.b_encoder_number_layers,
            b_encoder_neurons= sett.neurons_layer,
            b_encoder_actf= sett.b_encoder_actf,
            b_encoder_init= sett.b_encoder_initializer ,
            b_encoder_regularizer= sett.b_encoder_regularizer ,

            trunk_input_shape = 2, ## the value is set to be 2
            trunk_output_shape = sett.neurons_layer,  ### Needs to be same as branch output shape
            t_number_layers = sett.t_number_layers,
            t_neurons_layer = sett.neurons_layer, ## The same as b_neurons_lays
            t_actf = sett.t_actf,
            t_init = sett.t_initializer,
            t_regularizer = sett.t_regularizer,

            t_encoder_layers= sett.t_encoder_number_layers,
            t_encoder_neurons= sett.neurons_layer,
            t_encoder_actf= sett.t_encoder_actf,
            t_encoder_init= sett.t_encoder_initializer ,
            t_encoder_regularizer= sett.t_encoder_regularizer,
                )

model = don.don_model(nn)

init_lr = sett.init_lr

optimizer = tf.keras.optimizers.Adam(init_lr)

loss_obj = tf.keras.losses.MeanSquaredError()

model.compile(
    optimizer = optimizer,
    loss_fn = loss_obj)


# Prepare the dataset.
percent_trunk =  percent_trunk
# batch_size = int(vtn*vxn*percent_trunk*len( re_train_list))
batch_size = sett.batch_size

dataset = tf.data.Dataset.from_tensor_slices((b_train,t_train, target_train))
dataset = dataset.shuffle(buffer_size=batch_size).batch(batch_size)

# batch_size = int(vtn*vxn* percent_trunk*len( re_val_list))

val_dataset = tf.data.Dataset.from_tensor_slices((b_val,t_val, target_val))
val_dataset = val_dataset.batch(batch_size)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                              patience=sett.reduce_patience, min_lr=1e-6, min_delta=1e-10, verbose=1)

# early_stop = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     min_delta=1e-8,
#     patience=500,
#     verbose=1,
#     restore_best_weights=True
# )

i=1


timestamp_don = datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_dir = os.path.join(model_dir, 'burgers1d_mdon'+timestamp_don+model_suffix)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

np.save(out_dir+'/middle_checkpoint_id_branch.npy', id_branch)

class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10000 == 0:  # or save after some epoch, each k-th epoch etc.
            middle_model_path = os.path.join(out_dir,"middle_checkpoint", "model_{}".format(epoch))
            # self.model.save("model_{}.hd5".format(epoch)) 
            self.model.save(middle_model_path)
            lr = tf.keras.backend.get_value(self.model.optimizer.lr)

            save_lr_out_dir = os.path.join(out_dir,"middle_checkpoint", "mid_lr_saved_models.txt")

            with open(save_lr_out_dir, "a") as f:
                f.write("Epoch {}: Learning Rate: {}\n".format(epoch, lr))

if case == 'Train':

    init_time = time.time()

    # Train the model on all available devices.
    # model.fit(dataset, validation_data=val_dataset, epochs= train_epochs,
    #              callbacks=[tf.keras.callbacks.TensorBoard("tb/train_logs"+str(i)),reduce_lr])#,early_stop])

    saver = CustomSaver()

    model.fit(dataset, validation_data=val_dataset, epochs= train_epochs,
                 callbacks=[reduce_lr,saver])#,early_stop])

    # model.fit(dataset, validation_data=val_dataset, epochs= train_epochs,
    #              callbacks=[reduce_lr])#,early_stop])

    end_time = time.time()
    train_time = end_time - init_time
    hrs = int(train_time//3600); rem_time = train_time - hrs*3600
    mins = int(rem_time//60); secs = int(rem_time%60)
    print('Training time: %d H %d M, %d S'%(hrs,mins,secs))

    model.save(out_dir,id_branch,)

    
if case == 'Predict':   
    model = loaded_model



train_loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
lrate = model.history.history['lr']
train_epoch = model.history.epoch
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
csv_filename = optuna_rank + 'model_history.csv'
df.to_csv(csv_filename, index=False)

### save figure for the loss. 
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(14,5),constrained_layout=True)
ax[0].plot(train_epoch,train_loss,label='train_loss',marker='v',markevery=128)
ax[0].plot(train_epoch,val_loss,label='val_loss',marker='s',markevery=128)

ax[0].set_yscale('log')
ax[0].set_title('Validation and Training losses in semi-log scale')
ax[0].legend()

ax[1].plot(train_epoch,lrate,label='LR',marker='p',markevery=128)
ax[1].set_yscale('log')

ax[1].set_title('Learning rate decay')

#plt.savefig(optuna_rank + "_learning rate")

id_t = np.argmin(np.abs(vt-1.6))
id_x = np.argmin(np.abs(vx-0.8))

o_res = model([b_test,t_test])

if  scaling is True:
    o_res = u_scaler.inverse_transform(o_res)

test = np.reshape(np.array(o_res),(len( re_test_list),vxn,vtn))

error = test - burgers_array_test

plot_bounds_1d(burgers_array_test[0],test[0],error[0], L_test, T_test, label1= re_test_list[0], vmin1=0, vmax1=0.5,name='low_re'+str(i))
plot_bounds_1d(burgers_array_test[-1],test[-1],error[-1], L_test, T_test, label1= re_test_list[-1], vmin1=0, vmax1=0.5, name='high_re'+str(i))

test[:,:,id_t]=1000
test[:,id_x,:]=1000
burgers_array_test[:,:,id_t]=1000
burgers_array_test[:,id_x,:]=1000
error[:,:,id_t]=1000
error[:,id_x,:]=1000

plot_spcaetime_1d(test,
                  T_test,L_test,label= re_test_list,
                  vmin=0,
                  vmax=0.5,                      
                  name='prediction'+str(i))

plot_spcaetime_1d(error,
                  T_test,L_test,
                  colormap='coolwarm',label= re_test_list,
                  vmin=-0.05,
                  vmax=0.05,
                  name='error'+str(i))

plot_spcaetime_1d(burgers_array_train,
                  T_train,L_train,label= re_train_list,
                  vmin=0,
                  vmax=0.5,    
                  name='train') 

plot_spcaetime_1d(burgers_array_test,
                  T_test,L_test,label= re_test_list,
                  vmin=0,
                  vmax=0.5,    
                  name='truth')

