from keras.models import Model
from keras.applications import mobilenet_v2
from keras.layers import Conv2D, Dropout, Dense
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam

def GetModel(img_width, img_height):
    pretrain_net = mobilenet_v2.MobileNetV2(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet')

    freeze_before = None
    if freeze_before:
        for layer in pretrain_net.layers:
            if layer.name == freeze_before:
                break
            else:
                layer.trainable = False

    x = pretrain_net.output
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Dropout(rate=0.2, name='extra_dropout1')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid', name='classifier')(x)

    model = Model(inputs=pretrain_net.input, outputs=x, name='mobilenetv2_spoof')
    learning_rate = 5e-5  # Set the learning rate to use
    model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['acc'])

    return model