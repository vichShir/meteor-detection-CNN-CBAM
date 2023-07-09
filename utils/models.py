import tensorflow as tf
from utils.layers import ChannelAttention, SpatialAttention

class VGGModel():

    def __init__(self, image_size, vgg_model='16'):
        self.image_size = image_size
        self.__vgg_model = vgg_model
    
    
    def __pick_vgg_version(self):
        if self.__vgg_model == '16':
            return tf.keras.applications.vgg16.VGG16
        elif self.__vgg_model == '19':
            return tf.keras.applications.vgg19.VGG19
        else:
            raise Exception("VGG model not recognized. Only supported VGG16 and VGG19.")
    
    
    def __create_backbone(self):
        vgg = self.__pick_vgg_version()(input_shape=self.image_size,
                                        include_top=False,
                                        weights='imagenet')
        vgg.trainable = False
        return vgg
    
    
    def create_basic_model(self,
                           n_classes=2,
                           n_layers=16,
                           activation='relu',
                           dropout=0.5,
                           regularizer=None):
        backbone = self.__create_backbone()
        
        x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(n_layers, activation=activation, kernel_regularizer=regularizer)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(n_classes, activation='softmax', kernel_regularizer=regularizer)(x)
        
        model = tf.keras.models.Model(inputs=backbone.input, outputs=x, name='vgg' + self.__vgg_model)
        return model
    
    
    def create_bachnorm_model(self,
                              place_between_activation=False,
                              n_classes=2,
                              n_layers=16,
                              activation='relu',
                              dropout=0.5,
                              regularizer=None):
        basic_model = self.create_basic_model(n_classes, n_layers, activation, dropout, regularizer)
        
        if not place_between_activation:
            bn_model = tf.keras.models.Sequential()
            
            # Block 1
            for layer in basic_model.layers[:3]:
                bn_model.add(layer)
            bn_model.add(tf.keras.layers.BatchNormalization())

            # Block 2
            for layer in basic_model.layers[3:6]:
                bn_model.add(layer)
            bn_model.add(tf.keras.layers.BatchNormalization())

            # Block 3
            for layer in basic_model.layers[6:10]:
                bn_model.add(layer)
            bn_model.add(tf.keras.layers.BatchNormalization())

            # Block 4
            for layer in basic_model.layers[10:14]:
                bn_model.add(layer)
            bn_model.add(tf.keras.layers.BatchNormalization())

            # Block 5
            for layer in basic_model.layers[14:18]:
                bn_model.add(layer)
            bn_model.add(tf.keras.layers.BatchNormalization())

            # Block 6
            for layer in basic_model.layers[18:]:
                bn_model.add(layer)
                
        else:
            for i, layer in enumerate(basic_model.layers):
                if i==0:
                    input = layer.input
                    x = input
                else:
                    if "conv" in layer.name:
                        layer.activation = tf.keras.activations.linear
                        x = layer(x)
                        x = tf.keras.layers.BatchNormalization()(x)
                        x = tf.keras.layers.Activation('relu')(x)
                    else:
                        x = layer(x)
            bn_model = tf.keras.models.Model(inputs=input, outputs=x, name='vgg' + self.__vgg_model)
        
        return bn_model
    
    
    def create_cbam_model(self,
                          n_classes=2,
                          n_layers=16,
                          activation='relu',
                          dropout=0.5,
                          regularizer=None):
        
        basic_model = self.create_basic_model(n_classes, n_layers, activation, dropout, regularizer)
        bn_model = tf.keras.models.Sequential()
            
        # Block 1
        for layer in basic_model.layers[:3]:
            bn_model.add(layer)
        bn_model.add(ChannelAttention(224, 7))
        bn_model.add(SpatialAttention(7))
        bn_model.add(tf.keras.layers.BatchNormalization())

        # Block 2
        for layer in basic_model.layers[3:6]:
            bn_model.add(layer)
        bn_model.add(ChannelAttention(112, 7))
        bn_model.add(SpatialAttention(7))
        bn_model.add(tf.keras.layers.BatchNormalization())

        # Block 3
        for layer in basic_model.layers[6:10]:
            bn_model.add(layer)
        bn_model.add(ChannelAttention(56, 7))
        bn_model.add(SpatialAttention(7))
        bn_model.add(tf.keras.layers.BatchNormalization())

        # Block 4
        for layer in basic_model.layers[10:14]:
            bn_model.add(layer)
        bn_model.add(ChannelAttention(28, 7))
        bn_model.add(SpatialAttention(7))
        bn_model.add(tf.keras.layers.BatchNormalization())

        # Block 5
        for layer in basic_model.layers[14:18]:
            bn_model.add(layer)
        bn_model.add(ChannelAttention(14, 7))
        bn_model.add(SpatialAttention(7))
        bn_model.add(tf.keras.layers.BatchNormalization())

        # Block 6
        for layer in basic_model.layers[18:]:
            bn_model.add(layer)
        
        return bn_model