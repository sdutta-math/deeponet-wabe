import tensorflow as tf
import numpy as np

class don_nn(tf.keras.models.Model):
    def __init__(self, l_factor, latent_dim, branch_input_shape, b_number_layers, b_neurons_layer, b_actf, b_init, b_regularizer, b_encoder_layers, b_encoder_neurons, b_encoder_actf, b_encoder_init, b_encoder_regularizer, trunk_input_shape, t_number_layers, t_neurons_layer, t_actf, t_init, t_regularizer, t_encoder_layers, t_encoder_neurons, t_encoder_actf, t_encoder_init, t_encoder_regularizer, **kwargs):
        super().__init__(**kwargs)
        
        if b_regularizer=='none':
            b_regularizer=eval('None')

        if t_regularizer=='none':
            t_regularizer=eval('None')   
            
        if b_encoder_regularizer=='none':
            b_encoder_regularizer=eval('None')  

        if t_encoder_regularizer=='none':
            t_encoder_regularizer=eval('None') 
            
        self.latent_dim = latent_dim
        self.l_factor = l_factor
            
        self.branch_encoder = self.Branch_Encoder(branch_input_shape, b_neurons_layer, b_encoder_layers, b_encoder_neurons, b_encoder_actf, b_encoder_init, b_encoder_regularizer)
        
        self.trunk_encoder = self.Trunk_Encoder(trunk_input_shape, t_neurons_layer, t_encoder_layers, t_encoder_neurons, t_encoder_actf, t_encoder_init, t_encoder_regularizer)

        branch_encoder_img_file = 'branch_encoder.png'
        tf.keras.utils.plot_model(self.branch_encoder, to_file=branch_encoder_img_file, show_shapes=True)

        trunk_encoder_img_file = 'trunk_encoder.png'
        tf.keras.utils.plot_model(self.trunk_encoder, to_file=trunk_encoder_img_file, show_shapes=True)
        
        self.branch = self.MLP_Branch(branch_input_shape, trunk_input_shape, l_factor*latent_dim, b_number_layers, b_neurons_layer, b_actf, b_init, b_regularizer)
        
        self.trunk = self.MLP_Trunk(branch_input_shape, trunk_input_shape, l_factor*latent_dim, t_number_layers, t_neurons_layer, t_actf, t_init, t_regularizer)
        
        self.b0 = tf.Variable(tf.zeros(latent_dim,dtype=tf.float64), shape=tf.TensorShape(latent_dim), name='b0', dtype=tf.float64)   
        
        branch_img_file = 'branch_network.png'
        tf.keras.utils.plot_model(self.branch, to_file=branch_img_file, show_shapes=True)

        trunk_img_file = 'trunk_network.png'
        tf.keras.utils.plot_model(self.trunk, to_file=trunk_img_file, show_shapes=True)

    def Branch_Encoder(self, input_shape, neurons_layer, encoder_layers, encoder_neurons, actf, init, regularizer):
        input_layer = tf.keras.layers.Input(input_shape)
        encoder = input_layer
        for e in range(encoder_layers):
            encoder = tf.keras.layers.Dense(encoder_neurons, activation=actf, kernel_initializer=init, kernel_regularizer=regularizer)(encoder)
            encoder = tf.keras.layers.Dropout(0.1)(encoder)
        output_encoder = tf.keras.layers.Dense(neurons_layer, activation=actf, kernel_initializer=init, kernel_regularizer=regularizer)(encoder)
        model = tf.keras.Model(input_layer,output_encoder)
        return model

    def Trunk_Encoder(self, input_shape, neurons_layer, encoder_layers, encoder_neurons, actf, init, regularizer):
        input_layer = tf.keras.layers.Input(input_shape)
        encoder = input_layer
        for e in range(encoder_layers):
            encoder = tf.keras.layers.Dense(encoder_neurons, activation=actf, kernel_initializer=init, kernel_regularizer=regularizer)(encoder)
            encoder = tf.keras.layers.Dropout(0.1)(encoder)            
        output_encoder = tf.keras.layers.Dense(neurons_layer, activation=actf, kernel_initializer=init,kernel_regularizer=regularizer)(encoder)
        model = tf.keras.Model(input_layer,output_encoder)
        return model
            
    def MLP_Branch(self, branch_input_shape, trunk_input_shape, output_shape, number_layers, neurons_layer, actf, init, regularizer):
        branch_input_layer = tf.keras.layers.Input(branch_input_shape,name='branch_input')
        trunk_input_layer = tf.keras.layers.Input(trunk_input_shape,name='trunk_input')
        x = branch_input_layer
        for i in range(number_layers):
            x = tf.keras.layers.Dense(neurons_layer,activation=actf,kernel_initializer=init,kernel_regularizer=regularizer,name='branch_hidden'+str(i))(x)
            ones = tf.ones_like(x)
            x1 = tf.keras.layers.Subtract(name='subtract_branch'+str(i))([ones,x])
            x2 = tf.keras.layers.Multiply(name='multiply_branch_encoder'+str(i))([x,self.branch_encoder(branch_input_layer)])
            x3 = tf.keras.layers.Multiply(name='multiply_trunk_encoder'+str(i))([x1,self.trunk_encoder(trunk_input_layer)])  
            x = tf.keras.layers.Add(name='add_product'+str(i))([x2,x3])
            x = tf.keras.layers.Dropout(0.1)(x)
        output_layer = tf.keras.layers.Dense(output_shape,kernel_initializer=init,kernel_regularizer=regularizer,name='output_branch')(x)
        model = tf.keras.Model([branch_input_layer,trunk_input_layer],output_layer)
        return model
    
    def MLP_Trunk(self, branch_input_shape, trunk_input_shape, output_shape, number_layers, neurons_layer, actf, init, regularizer):
        branch_input_layer = tf.keras.layers.Input(branch_input_shape,name='branch_input')
        trunk_input_layer = tf.keras.layers.Input(trunk_input_shape,name='trunk_input')
        x = trunk_input_layer
        for i in range(number_layers):
            x = tf.keras.layers.Dense(neurons_layer,activation=actf,kernel_initializer=init,kernel_regularizer=regularizer,name='branch_hidden'+str(i))(x)
            ones = tf.ones_like(x)
            x1 = tf.keras.layers.Subtract(name='subtract_branch'+str(i))([ones,x])
            x2 = tf.keras.layers.Multiply(name='multiply_branch_encoder'+str(i))([x,self.branch_encoder(branch_input_layer)])
            x3 = tf.keras.layers.Multiply(name='multiply_trunk_encoder'+str(i))([x1,self.trunk_encoder(trunk_input_layer)])  
            x = tf.keras.layers.Add(name='add_product'+str(i))([x2,x3])
            x = tf.keras.layers.Dropout(0.1)(x)
        output_layer = tf.keras.layers.Dense(output_shape, activation=actf, kernel_initializer=init,kernel_regularizer=regularizer,name='output_trunk')(x)

        model = tf.keras.Model([branch_input_layer,trunk_input_layer],output_layer)
        return model
        
    def call(self,data):
        br = tf.reshape(self.branch(data), [-1, self.latent_dim, self.l_factor])# automate the m and p shapes
        tr = tf.reshape(self.trunk(data), [-1, self.latent_dim, self.l_factor])
        onet = tf.einsum('ijk,ijk->ij', br, tr) + self.b0
#         onet = tf.reduce_sum(self.branch(data)*self.trunk(data), axis=1, keepdims=True) + self.b0
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