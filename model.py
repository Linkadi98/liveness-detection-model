from keras import Model
from keras.src.applications.mobilenet_v3 import MobileNetV3Small, MobileNetV3Large
from keras.src.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, AveragePooling2D, \
    Flatten
from keras.src.optimizers import Adam


def GetModel(img_width, img_height, type='small'):
    # Load MobileNetV3 Large as the backbone
    if type == 'small':
        pretrain_net = MobileNetV3Small(
            input_shape=(img_width, img_height, 3),
            include_top=False,
            weights='imagenet'
        )
    else:
        pretrain_net = MobileNetV3Large(
            input_shape=(img_width, img_height, 3),
            include_top=False,
            weights='imagenet'
        )

    for layer in pretrain_net.layers:  # Freeze first 100 layers
        layer.trainable = True

    x = pretrain_net.output

    p = 0.6

    x = BatchNormalization(axis=1, name="net_out")(x)
    x = Dropout(p / 4)(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Dropout(p)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Dropout(p / 2)(x)

    # Final classification layer
    x = Dense(1, activation='sigmoid', name='classifier')(x)

    # Compile the model
    model = Model(inputs=pretrain_net.input, outputs=x, name='mobilenetv3_spoof')
    learning_rate = 1e-4
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['acc'])

    return model
