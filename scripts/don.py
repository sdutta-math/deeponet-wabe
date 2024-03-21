import tensorflow as tf

if tf.__version__ == '2.16.1':
    import tf_keras as keras
else:
    import tensorflow.keras as keras

import numpy as np

class don_nn(keras.models.Model):
    def __init__(self, branch_input_shape, branch_output_shape, b_number_layers, b_neurons_layer, b_actf, b_init, b_regularizer, 
                 trunk_input_shape, trunk_output_shape, t_number_layers, t_neurons_layer, t_actf, t_init, t_regularizer,
                 dropout=False, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        assert branch_output_shape == trunk_output_shape, print(f"Branch and Trunk output shape needs to be the same.")
        
        self.dropout = dropout
        self.droput_rate = dropout_rate
        
        if b_regularizer=='none':
            b_regularizer=eval('None')

        if t_regularizer=='none':
            t_regularizer=eval('None')
        
        self.branch = self.MLP_Branch(branch_input_shape, branch_output_shape, b_number_layers, b_neurons_layer, b_actf, b_init, b_regularizer)
        
        self.trunk = self.MLP_Trunk(trunk_input_shape, trunk_output_shape, t_number_layers, t_neurons_layer, t_actf, t_init, t_regularizer)
        
        self.b0 = tf.Variable(0, name='b0', dtype=tf.float64)    
    
    def MLP_Branch(self, input_shape, output_shape, number_layers, neurons_layer, actf, init, regularizer):
        input_layer = keras.layers.Input(input_shape)
        x = input_layer
        for i in range(number_layers):
            x = keras.layers.Dense(neurons_layer,activation=actf,kernel_initializer=init,kernel_regularizer=regularizer)(x)
            if self.dropout==True: 
                x = keras.layers.Dropout(self.dropout_rate)(x)
        output_layer = keras.layers.Dense(output_shape,kernel_initializer=init,kernel_regularizer=regularizer)(x)
        model = keras.Model(input_layer,output_layer)
        return model
    
    def MLP_Trunk(self, input_shape, output_shape, number_layers, neurons_layer, actf, init, regularizer):
        input_layer = keras.layers.Input(input_shape)
        x = input_layer
        for i in range(number_layers):
            x = keras.layers.Dense(neurons_layer,activation=actf,kernel_initializer=init,kernel_regularizer=regularizer)(x)
            if self.dropout==True: 
                x = keras.layers.Dropout(self.dropout_rate)(x)
        output_layer = keras.layers.Dense(output_shape,activation=actf,kernel_initializer=init,kernel_regularizer=regularizer)(x)
        model = keras.Model(input_layer,output_layer)
        return model
        
    def call(self,data):
        b, t = data
        onet = tf.reduce_sum(self.branch(b)*self.trunk(t), axis=1, keepdims=True) + self.b0
        return onet
    
class don_model(keras.Model):
    def __init__(self, model):
        super(don_model, self).__init__()
        self.model = model

    def compile(self, optimizer, loss_fn):
        super(don_model, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def save(self,path,id_b=None):
        keras.models.save_model(self.model,path)
        
        if id_b is not None:
            np.save(path+'/branch_id',id_b)
        
    def call(self, dataset):

        return self.model(dataset)
    
    def train_step(self, dataset):
        
        self.b_input, self.t_input, self.target = dataset
        
        with tf.GradientTape() as tape:
            o_res = self.model([self.b_input,self.t_input], training=True)
            loss = self.loss_fn(o_res,self.target)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return {"loss": loss}

    def test_step(self, dataset):

        self.b_input, self.t_input, self.target = dataset

        val = self.model([self.b_input,self.t_input], training=False)
        val_loss = self.loss_fn(val,self.target)

        return {"loss": val_loss}
