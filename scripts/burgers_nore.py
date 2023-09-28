import json
import os
import sys
import argparse
import tensorflow as tf
import keras_tuner
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

sys.path.append('/home/pgrivera/DeepONet/deeponet/Burgers/functions')
sys.path.append('/home/pgrivera/DeepONet/scripts')

import don
import burgers_exact as bg


parser = argparse.ArgumentParser(description='DeepONet')
parser.add_argument('-tuner_epochs', default=1000, type=int, help='Number of tuning epochs')
parser.add_argument('-train_epochs', default=10000, type=int, help='Number of training epochs')
parser.add_argument('-loss', default='mse', action='store', type=str, help='Loss Function')
parser.add_argument('-scaling', action='store_true')
parser.add_argument('-scaler_min', default=-1, type=int, help='Lower bound for scaler')
parser.add_argument('-scaler_max', default=1, type=int, help='Uper bound for scaler')
parser.add_argument('-re_train_list', nargs='+', type=int)
parser.add_argument('-re_val_list', nargs='+', type=int)
parser.add_argument('-re_test_list', nargs='+', type=int)
parser.add_argument('-tuner', default='random', action='store', type=str, help='Search Algorithm')
parser.add_argument('-x_extent_train', default=1, type=float, help='Uper bound for x')
parser.add_argument('-t_extent_train', default=2, type=float, help='Uper bound for t')
parser.add_argument('-x_extent_val', default=1, type=float, help='Uper bound for x')
parser.add_argument('-t_extent_val', default=2, type=float, help='Uper bound for t')
parser.add_argument('-percent_branch', default=0.05, type=float, help='Percent branch sensors')
parser.add_argument('-percent_trunk', default=0.05, type=float, help='Percent trunk sensors')

args = parser.parse_args()


original_stdout = sys.stdout

sys.stdout = open("arguments.txt", "w")

print(args)

sys.stdout = original_stdout # Reset the standard output to its original value

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
                      vmin=-0.01, vmax=0.01, extent=(0,T,0,L), aspect = 0.79)
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
Re_train = args.re_train_list
L_train = args.x_extent_train
T_train = args.t_extent_train
vxn = 300
vtn = 500
vx, vt = define_grid(L_train, T_train, vxn, vtn)
percent_branch = args.percent_branch
percent_trunk = args.percent_trunk
burgers_array_train, burgers_flatten_train, b_train, \
                    t_train, target_train, id_branch = multiple_burgers(Re_train,vxn,vx,vtn,vt,
                                                            percent_branch,percent_trunk)

Re_val = args.re_val_list
L_val = args.x_extent_val
T_val = args.t_extent_val
vxn = 300
vtn = 500
vx, vt = define_grid(L_val, T_val, vxn, vtn)
percent_branch = args.percent_branch
percent_trunk = args.percent_trunk
burgers_array_val, burgers_flatten_val, b_val, \
                    t_val, target_val, _ = multiple_burgers(Re_val,vxn,vx,vtn,vt,
                                                         percent_branch,percent_trunk,id_branch=id_branch)

Re_test = args.re_test_list
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

if args.scaling is True:
    x_scaler = MinMaxScaler(feature_range=(args.scaler_min, args.scaler_max))
    t_scaler = MinMaxScaler(feature_range=(args.scaler_min, args.scaler_max))
    u_scaler = MinMaxScaler(feature_range=(args.scaler_min, args.scaler_max))
    b_scaler = MinMaxScaler(feature_range=(args.scaler_min, args.scaler_max))

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

branch_sensors = int(vxn*args.percent_branch)
    
def build_model(hp):
    nn = don.don_nn(branch_sensors, 
                    2, 
                    hp.Int("number_layers", min_value=1, max_value=6, step=1),
                    hp.Int("number_neurons", min_value=16, max_value=256, step=16),
                    hp.Choice("activation", ["relu", "elu", "tanh", "sigmoid"]),
                    hp.Choice("initializer", ["glorot_normal", "he_normal"]),                
                    hp.Choice("regularizer", ["l1", "l2", "none"]))
                
    model = don.don_model(nn)
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                              patience=100, min_lr=1e-6, min_delta=1e-10, verbose=1)
    
    optimizer_str = hp.Choice("optimizer", ["adam", "rmsprop"])
    if optimizer_str == "adam":
        optimizer = tf.keras.optimizers.Adam()
    elif optimizer_str == "rmsprop":   
        optimizer = tf.keras.optimizers.RMSprop()
        
    if args.loss == "mse":
        loss_obj = tf.keras.losses.MeanSquaredError()
    elif args.loss == "mae":   
        loss_obj = tf.keras.losses.MeanAbsoluteError()
        
    model.compile(
        optimizer = optimizer,
        loss_fn = loss_obj
    )
    return model

if args.tuner == "random":

    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective="val_loss",
        max_trials=100,
        executions_per_trial=1,
        overwrite=True,
        directory="Tuner",
        project_name="Burgers",
    )

elif args.tuner == "bayesian":

    tuner = keras_tuner.BayesianOptimization(
        hypermodel=build_model,
        objective="val_loss",
        max_trials=100,
        executions_per_trial=1,
        beta=5,
        overwrite=True,
        directory="Tuner",
        project_name="Burgers",
    )

# Prepare the dataset.
percent_trunk = args.percent_trunk
batch_size = int(vtn*vxn*percent_trunk*len(args.re_train_list))

dataset = tf.data.Dataset.from_tensor_slices((b_train,t_train, target_train))
dataset = dataset.shuffle(buffer_size=batch_size).batch(batch_size)

batch_size = int(vtn*vxn*args.percent_trunk*len(args.re_val_list))

val_dataset = tf.data.Dataset.from_tensor_slices((b_val,t_val, target_val))
val_dataset = val_dataset.batch(batch_size)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                              patience=100, min_lr=1e-6, min_delta=1e-10, verbose=1)

tuner.search(dataset, validation_data=val_dataset, epochs=args.tuner_epochs,
             callbacks=[tf.keras.callbacks.TensorBoard("tb/tuner_logs"),reduce_lr])

original_stdout = sys.stdout

sys.stdout = open("tuner.txt", "w")

print(tuner.search_space_summary())
print(tuner.results_summary())

sys.stdout = original_stdout # Reset the standard output to its original value

# Get the top hyperparameters.
best_hps = tuner.get_best_hyperparameters(5)
      
for i in range(5):
    # Build the model with the best hp.
    model = build_model(best_hps[i])
    model.fit(dataset, validation_data=val_dataset, epochs=args.train_epochs,
              callbacks=[tf.keras.callbacks.TensorBoard("tb/train_logs"), reduce_lr])

    model.save('tuner_test'+str(i),id_branch)
    
    o_res = model([b_test,t_test])
    
    if args.scaling is True:
        o_res = u_scaler.inverse_transform(o_res)

    test = np.reshape(np.array(o_res),(len(args.re_test_list),vxn,vtn))

    error = test - burgers_array_test

    plot_spcaetime_1d(test[0],test[1],
                      test[2],test[3],
                      test[4],test[5],
                      T_test,L_test,label1=args.re_test_list,
                      vmin1=0,
                      vmax1=0.5,                      
                      name='prediction'+str(i))
    
    plot_spcaetime_1d(error[0],error[1],
                      error[2],error[3],
                      error[4],error[5],
                      T_test,L_test,
                      colormap='coolwarm',label1=args.re_test_list,
                      vmin1=-0.01,
                      vmax1=0.01,
                      name='error'+str(i))

    plot_bounds_1d(burgers_array_test[0],test[0],error[0], L_test, T_test, label1=args.re_test_list[0], vmin1=0, vmax1=0.5,name='low_re'+str(i))
    plot_bounds_1d(burgers_array_test[-1],test[-1],error[-1], L_test, T_test, label1=args.re_test_list[-1], vmin1=0, vmax1=0.5, name='high_re'+str(i))


      
plot_spcaetime_1d(burgers_array_train[0],burgers_array_train[1],
                  burgers_array_train[2],burgers_array_train[3],
                  burgers_array_train[4],burgers_array_train[5],
                  T_train,L_train,label1=args.re_train_list,
                  vmin1=0,
                  vmax1=0.5,                  
                  name='train') 

plot_spcaetime_1d(burgers_array_test[0],burgers_array_test[1],
                  burgers_array_test[2],burgers_array_test[3],
                  burgers_array_test[4],burgers_array_test[5],
                  T_test,L_test,label1=args.re_test_list,
                  vmin1=0,
                  vmax1=0.5,    
                  name='truth')



    