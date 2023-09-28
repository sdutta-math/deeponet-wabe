import tensorflow as tf
import numpy as np

class don_nn(tf.keras.models.Model):
    def __init__(self, branch_input_shape, trunk_input_shape, number_layers, neurons_layer, actf, init, regularizer, **kwargs):
        super().__init__(**kwargs)
        
        if regularizer=='none':
            regularizer=eval('None')
        
        self.branch = self.MLP_Branch(branch_input_shape, neurons_layer, number_layers, neurons_layer, actf, init, regularizer)
        
        self.trunk = self.MLP_Trunk(trunk_input_shape, neurons_layer, number_layers, neurons_layer, actf, init, regularizer)
        
        self.b0 = tf.Variable(0, name='b0', dtype=tf.float64)    
    
    def MLP_Branch(self, input_shape, output_shape, number_layers, neurons_layer, actf, init, regularizer):
        input_layer = tf.keras.layers.Input(input_shape)
        x = input_layer
        for i in range(number_layers):
            x = tf.keras.layers.Dense(neurons_layer,activation=actf,kernel_initializer=init,kernel_regularizer=regularizer)(x)
        output_layer = tf.keras.layers.Dense(output_shape,kernel_initializer=init,kernel_regularizer=regularizer)(x)
        model = tf.keras.Model(input_layer,output_layer)
        return model
    
    def MLP_Trunk(self, input_shape, output_shape, number_layers, neurons_layer, actf, init, regularizer):
        input_layer = tf.keras.layers.Input(input_shape)
        x = input_layer
        for i in range(number_layers):
            x = tf.keras.layers.Dense(neurons_layer,activation=actf,kernel_initializer=init,kernel_regularizer=regularizer)(x)
        output_layer = tf.keras.layers.Dense(output_shape,activation=actf,kernel_initializer=init,kernel_regularizer=regularizer)(x)
        model = tf.keras.Model(input_layer,output_layer)
        return model
        
    def call(self,data):
        b, t = data
        onet = tf.reduce_sum(self.branch(b)*self.trunk(t), axis=1, keepdims=True) + self.b0
        return onet
    
class don_model(tf.keras.Model):
    def __init__(self, model):
        super(don_model, self).__init__()
        self.model = model

    def compile(self, optimizer, loss_fn):
        super(don_model, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def save(self,path,id_b=None):
        tf.keras.models.save_model(self.model,path)
        
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