import tensorflow as tf
from tensorflow import keras as k
from keras_tuner import HyperModel
from matplotlib import pyplot as plt

# TODO Keep tuning using Keras Tuner and TensorBoard

class NNmodel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
    def build(self, hp):

        # Hyperparameter search space
        learning_rate = hp.Float('learning_rate',min_value=1e-6,max_value=1e-5,default=5e-6,sampling='linear')
        clipnorm= hp.Float('clipnorm',min_value=0.5,max_value=10.0,default=1.0)
        clipvalue= hp.Float('clipvalue',min_value=0.1,max_value=0.3,default=0.2)
        #     # Initial hidden layers
        units_i= hp.Int('units_i',min_value=10,max_value=1000,default=70,sampling='log')
        # activation_i=hp.Choice('hidden_activation_i',values=['relu', 'tanh', 'softmax'],default='relu')
        dropout_i= hp.Float('dropout_i',min_value=0.1,max_value=0.8,default=0.3)
        # l2regularization_i= hp.Float('l2regularization_i',min_value=0.0001,max_value=0.1,sampling='log')
        # gaussianNoise_i= hp.Float('gaussianNoise_i',min_value=0.001,max_value=2,sampling='log')
        batch_norm = hp.Boolean('bacht_norm')
        #     # Intermediate hidden layers
        # max_value_ihl = 0
        # num_ihl= hp.Int('num_intermediate_hidden_layers',min_value=0,max_value=max_value_ihl,default=1)
        # units= hp.Int('units',min_value=250,max_value=400,default=140,sampling='linear')
        # activation=hp.Choice('hidden_activation',values=['relu', 'tanh', 'softmax'],default='relu')
        # dropout= hp.Float('dropout',min_value=0.0001,max_value=0.5,default=0.3,sampling='log')
        # l2regularization= hp.Float('l2regularization',min_value=0.0001,max_value=0.1,sampling='log')
        # gaussianNoise = hp.Float('gaussianNoise',min_value=0.001,max_value=2.0,sampling='log')
        #     # Final hidden layers
        units_f= hp.Int('units_f',min_value=10,max_value=1000,default=400,sampling='log')
        # activation_f=hp.Choice('hidden_activation_f',values=['relu', 'tanh', 'softmax'],default='relu')
        dropout_f= hp.Float('dropout_f',min_value=0.2,max_value=0.7,default=0.3)
        # l2regularization_f= hp.Float('l2regularization_f',min_value=0.0001,max_value=0.1,sampling='log')
        # gaussianNoise_f = hp.Float('gaussianNoise_f',min_value=0.001,max_value=2.0,sampling='log')
        
        # Model
        model = k.Sequential()
            # Sequential() infers the input layer
            
            # Initial hidden layers

        model.add( k.layers.Dense( units_i))
        model.add( k.layers.Dropout( dropout_i ) )

        # model.add(
        #     k.layers.Dense(
        #         units=units_i,
        #         activation=activation_i,
        #         activity_regularizer= k.regularizers.l2(l2regularization_i)
        #     )
        # )
        # model.add( k.layers.Dropout( dropout_i ) )
        if batch_norm == True:
            model.add( k.layers.BatchNormalization() )
        # model.add( k.layers.GaussianNoise( gaussianNoise_i ) )
        
            # Intermediate hidden layers
        
        # for i in range( num_ihl ):
        #     with hp.conditional_scope( 'num_intermediate_hidden_layers', list(range(i+1 , max_value_ihl+1 )) ):
        #         model.add(
        #             k.layers.Dense(
        #                 units=hp.Int("units_" + str(i+1), min_value=32, max_value=512, step=32),
        #                 activation='relu',
        #                 # activity_regularizer= k.regularizers.l2(l2regularization)
        #             )
        #         )
        #         model.add( k.layers.Dropout( dropout, name='dropout_' + str(i+1) ) )
        #         model.add( k.layers.BatchNormalization() )
        #         model.add( k.layers.GaussianNoise( gaussianNoise ) )
        

            # Final hidden layers

        model.add( k.layers.Dense(units_f))
        model.add( k.layers.Dropout( dropout_f ) )

        # model.add(tf.keras.layers.Reshape((-1,1)))
        # model.add( k.layers.LSTM(16))
        # # model.add( k.layers.GRU(16))
        # # model.add( k.layers.SimpleRNN(16))

        # model.add(
        #     k.layers.Dense(
        #         units=units_f,
        #         activation=activation_f,
        #         activity_regularizer= k.regularizers.l2(l2regularization_f)
        #     )
        # )
        # model.add( k.layers.Dropout( dropout_f ) )
        # model.add( k.layers.GaussianDropout( 0.5 ) )
        # model.add( k.layers.ActivityRegularization(l1=0.1, l2=0.1 ) )
        # model.add( k.layers.LayerNormalization() )
        # model.add( k.layers.BatchNormalization() )
        # model.add( k.layers.GaussianNoise( gaussianNoise_f ) )
        
            # Output layer
        model.add( k.layers.Dense(self.num_classes, activation='softmax'))
            # Compile
        loss_fn= k.losses.CategoricalCrossentropy(name='loss')              
        model.compile(
            optimizer = k.optimizers.Adam(learning_rate=learning_rate,clipnorm=clipnorm,clipvalue=clipvalue),
            loss = loss_fn,
            metrics = [acurracy],
            # metrics = [ acurracy, recall, precission, sensatspecf, specfatsens, auc_roc, auc_pr ]
        )
        return model
# Classification metrics
acurracy = k.metrics.CategoricalAccuracy(name='acurracy')
# Plot model history
def plotHistory(history):
    fig, ( (ax1, ax2) ) = plt.subplots(2, 1, sharex=True, figsize= (7,7))
    # fig, ( (ax1, ax2), (ax3 , ax4), (ax5,ax6), (ax7,ax8)) = plt.subplots(4, 2, sharex=True, figsize= (10,10))
    fig.text(0.5, 0.05, 'Epochs', ha='center')
    x = range( 1, len(history.history['loss'])+1 )
    ax1.plot( x, history.history['loss'], label='train' )
    ax1.plot( x, history.history['val_loss'], label='validation' )
    ax1.set_title('Loss function')
    ax2.plot( x, history.history['acurracy'], label='train' )
    ax2.plot( x, history.history['val_acurracy'], label='validation' )
    ax2.set_title('CategoricalAcurracy')
    plt.legend()
    plt.show()
# TODO Define custom metrics for one-hot-encoded predictions and labels like in this example:
# recall, precission, f1, precission at recall, specifity at sensitivity, auc of the roc curve, ...
