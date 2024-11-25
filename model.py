from keras.layers import Conv2D, Dropout, Dense
from keras.layers import GlobalAveragePooling2D
from keras.optimizers.legacy import Adam
from keras.models import Model
from keras.src.applications import MobileNetV3Large
from keras.src.applications import MobileNetV3Small


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

    for layer in pretrain_net.layers:
        layer.trainable = True  # Allow all layers to be trainable

    # Add custom classification layers
    x = pretrain_net.output
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(rate=0.2)(x)

    # Final classification layer
    x = Dense(1, activation='sigmoid', name='classifier')(x)

    # Compile the model
    model = Model(inputs=pretrain_net.input, outputs=x, name='mobilenetv3_spoof')
    learning_rate = 0.0001
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['acc'])

    return model
