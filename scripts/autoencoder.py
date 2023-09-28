#! /usr/bin/env python

import numpy as np
import scipy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, LSTM, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers


def gen_batch_ae(X_train, X_val, batch_size = None, shuffle_buffer = 1000):
    """
    Utility function to create a minibatch generator
    for Autoencoder training
    using tensorflow.data.dataset module
    """
    X_train = tf.convert_to_tensor(X_train,dtype=tf.float64)
    X_val = tf.convert_to_tensor(X_val,dtype=tf.float64)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train,)).shuffle(shuffle_buffer)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, ))
    if batch_size is not None:
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
    
    return train_dataset, val_dataset



class Encoder(tf.keras.layers.Layer):

    def __init__(self, latent_dim, act, hidden_dim, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, )

        self.l2 = regularizers.l2(kwargs['regu'])
        self.latent_dim = latent_dim
        self.activation = act
        self.hidden_units = hidden_dim[:]
        self.input_dim = hidden_dim[0]
        self.input_layer  = InputLayer(input_shape=(self.input_dim,))
        self.output_layer = Dense(self.latent_dim, activation='linear')
        self.hidden_layers = [Dense(u,activation=self.activation,kernel_regularizer=self.l2) for u in self.hidden_units[1:]]

    def call(self, input_features):
        x = self.input_layer(input_features)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
    
    def get_config(self):
        return {"hidden_units": self.hidden_units,
                "activation": self.activation,
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "l2": self.l2}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Decoder(tf.keras.layers.Layer):

    def __init__(self, latent_dim, act, hidden_dim, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, )

        self.l2 = regularizers.l2(kwargs['regu'])
        
        self.activation = act
        self.hidden_units = np.flip(hidden_dim)
        self.input_dim = latent_dim      
        self.output_dim = self.hidden_units[-1]
            
        self.input_layer  = InputLayer(input_shape=(self.input_dim,))
        self.output_layer = Dense(self.output_dim, activation='linear')
        self.hidden_layers = [Dense(u,activation=self.activation,kernel_regularizer=self.l2) for u in self.hidden_units[:-1]]

    def call(self, latent_code):
        x = self.input_layer(latent_code)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def get_config(self):
        return {"hidden_units": self.hidden_units,
                "activation": self.activation,
                "input_dim": self.input_dim,
                "output_dim":self.output_dim,
                "l2": self.l2}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
class Autoencoder(tf.keras.Model):

    def __init__(self, latent_dim, enc_act, dec_act, hidden_dim, name="autoencoder", **kwargs):
        super(Autoencoder, self).__init__(name=name, **kwargs)

        try:
            self.l2_lam = kwargs['l2_lam']
        except:
            self.l2_lam = 1e-6
        self.reg_wt = 0.01
        
        
        self.latent_dim = latent_dim
        self.enc_activation = enc_act
        self.dec_activation = dec_act
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(self.latent_dim, self.enc_activation, self.hidden_dim,regu=self.l2_lam)
        self.decoder = Decoder(self.latent_dim, self.dec_activation, self.hidden_dim,regu=self.l2_lam)

    def call(self, input_features):
        encoded = self.encoder(input_features)
        pred = self.decoder(encoded)
        return pred
    
    def compile(self, optimizer, loss_fn):
        super(Autoencoder, self).compile(loss=loss_fn, optimizer=optimizer,)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    def get_config(self):
        return {"hidden_units": self.hidden_dim,
                "encoder_activation": self.enc_activation,
                "decoder_activation": self.dec_activation,
                "latent_dim": self.latent_dim}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def train_step(self, train_ds, ):
        
        # Unpack the data        
        if isinstance(train_ds, tuple):
            batch_train = train_ds[0]
        else:
            batch_train = train_ds

#         tf.print(batch_train[0])

        
        with tf.GradientTape() as tape:
            ## Added by SD to save model
            decoded_tmp = self(batch_train, training=True)
           
            ## Encoded output for this minibatch
            encoded = self.encoder(batch_train, training=True)
            ## Forward Evaluation for this minibatch
            decoded = self.decoder(encoded, training=True)   
            ## L2 Regularization Loss Component
            l2_loss=tf.add_n(self.losses) #self.encoder.losses

            ## Compute the loss value for this minibatch.
            loss = self.loss_fn(batch_train, decoded,)              
            reg_loss = self.reg_wt*tf.sqrt(l2_loss)
            loss += reg_loss
            
        trainable_vars = self.encoder.trainable_variables \
                         + self.decoder.trainable_variables
        grads = tape.gradient(loss, trainable_vars)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
#         epoch_loss_metric.update_state(loss)
        return {"loss": loss}
    
    
    ## Customize model.evaluate() calls using test_step()
    def test_step(self, val_ds, ):
        
        # Unpack the data
        if isinstance(val_ds, tuple):
            val_batch_test = val_ds[0]
        else:
            val_batch_test = val_ds
         

        val_encoded = self.encoder(val_batch_test, training=False)
        val_decoded = self.decoder(val_encoded, training=False)  
        val_loss = self.loss_fn(val_batch_test, val_decoded,)
        
#         epoch_val_loss_metric.update_state(val_loss)
        return {"loss": val_loss}



class Optimizer:
    def __init__(self, lr, clip=None):
        self.lr=lr
        print(self.lr)
#         self.clip = clip

    def get_opt(self, opt):
        """Dispatch method"""
        method_name = 'opt_' + str(opt)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid optimizer")
        # Call the method as we return it
        return method()

    def opt_Adam(self):
        return tf.keras.optimizers.Adam(learning_rate=self.lr, ) #clipvalue=self.clip

    def opt_SGD(self):
        return  tf.keras.optimizers.SGD(learning_rate=self.lr, )
    
    def opt_RMSprop(self):
        return  tf.keras.optimizers.RMSprop(learning_rate=self.lr, )
    
    def opt_Adamax(self):
        return  tf.keras.optimizers.Adamax(learning_rate=self.lr, )
    
    def opt_Adagrad(self):
        return  tf.keras.optimizers.Adagrad(learning_rate=self.lr, )

    
## Custom Loss Function using keras loss class
class MyNMSELoss(tf.keras.losses.Loss):
    # initialize instance attributes
    def __init__(self, ):
        super(MyNMSELoss, self).__init__()
        ## The wrapper allows additional arguments
        ## to be passed to the loss function
        # self.threshold = threshold 
        
    # Compute loss
    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_pred - y_true))
        true_norm = tf.reduce_mean(tf.square(y_true)) + 1e-6
        return  tf.truediv(mse, true_norm)
#         loss.__name__ = "normalized_mean_squared_error_loss"



    
def save_model(u_model, input_shape, results):
    ## To use TF SavedModel format with a custom training loop,
    ## call model.predict() on some input tensors first.
    ## Otherwise TF doesn't know the shape and dtype of input data
    ## it should be expecting, and thus cannot create it's weight 
    ## variables. When using model.fit() this step happens automatically.
    
    u_model.build(input_shape)
    
    u_model.save(results['savedir']+'/autoenc')

    np.savez_compressed(results['savedir']+'/model_history', 
                        loss = results['loss'], valloss = results['valloss'], 
                        epochs = results['epochs'], 
                        umax = results['umax'], umin = results['umin'],
                        msg=results['msg'], timestamp=results['timestamp'])
    
    
def load_model(savedir):
    ## When using custom loss functions while training, there are two ways
    ## to load a saved model
    ## 1) If loaded model will not be used for retraining, then
    ##   use 'compile=False' option while loading so that TF does
    ##   search for custom objects loss functions
    u_model = tf.keras.models.load_model(savedir+'/autoenc', compile=False,
#                                            custom_objects={"autoencoder": autoencoder,
#                                                            "Encoder": Encoder,
#                                                            "Decoder": Decoder}
                                        )
    results = np.load(savedir+'/model_history.npz')

    return u_model, results
