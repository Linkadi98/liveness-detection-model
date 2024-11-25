from keras import Model
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2


def getModel(img_width, img_height, type='small'):
    # Load MobileNetV3 Small/Large as the backbone
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

    for layer in pretrain_net.layers[150:]:
        layer.trainable = False

    # Add custom classification layers
    x = pretrain_net.output
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)  # Normalize activation
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # Reduce overfitting
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Final classification layer
    x = Dense(1, activation='sigmoid', name='classifier')(x)

    # Compile the model
    model = Model(inputs=pretrain_net.input, outputs=x, name='mobilenetv3_spoof')
    learning_rate = 1e-4
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['acc']
    )
    return model
