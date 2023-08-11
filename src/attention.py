from keras.layers import Layer
from keras.layers import dot
import keras.backend as K


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weights',
                                 shape=(input_shape[0][-1], input_shape[0][-1]),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        attention = K.dot(inputs[1], self.W)
        attention = K.softmax(attention, axis=-1)
        weighted_sum = dot([attention, inputs[0]], axes=[-1, -1])
        return [weighted_sum, attention]
