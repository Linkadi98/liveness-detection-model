import os
import glob
import pandas as pd
import tensorflow as tf
from model import GetModel
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

img_width = 224
img_height = 224

seed_number = 24

train_batch_size = 32
val_batch_size = 32
batch_size = 1

num_epochs = 15
validation_steps = 1

if __name__ == '__main__':

    model = GetModel(img_width, img_height)

    input_dir = "data"
    train_dir = os.path.join(input_dir, 'training')
    val_dir = os.path.join(input_dir, 'development')
    test_dir = os.path.join(input_dir, 'evaluation')
    label_name = [subdir for subdir in sorted(os.listdir(train_dir)) if os.path.isdir(os.path.join(train_dir, subdir))]

    dir_dict = {'train': train_dir, 'val': val_dir, 'test': test_dir}
    case_count, set_length = {}, {}

    for key, val in dir_dict.items():
        case_count[key] = {}
        set_count = 0

        for label in label_name:
            label_list = list(sorted(glob.glob(os.path.join(val, label, "*.png"))))
            if len(label_list) == 0:
                continue

            case_count[key][label] = len(label_list)
            set_count += len(label_list)

        set_length[key] = set_count

    case_count_df = pd.DataFrame(case_count)

    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255) if set_length["test"] > 0 else None
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.15,
                                       zoom_range=0.15,
                                       horizontal_flip=True,
                                       fill_mode="nearest")



    train_gen = train_datagen.flow_from_directory(train_dir,
                                                  batch_size=train_batch_size,
                                                  class_mode='binary',
                                                  target_size=(img_width, img_height),
                                                  seed=seed_number)

    val_gen = val_datagen.flow_from_directory(val_dir,
                                              batch_size=val_batch_size,
                                              class_mode='binary',
                                              target_size=(img_width, img_height),
                                              seed=seed_number)

    if test_datagen is not None:
        test_gen = test_datagen.flow_from_directory(test_dir,
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    target_size=(img_width, img_height),
                                                    seed=seed_number,
                                                    shuffle=False)
    else:
        test_gen = None

    train_length = len(train_gen.classes)
    weight0 = train_length / case_count_df['train'][label_name[0]] * (1 / len(label_name))
    weight1 = train_length / case_count_df['train'][label_name[1]] * (1 / len(label_name))
    class_weight = {0: weight0, 1: weight1}

    train_id = "train-weights"
    save_dir = os.path.join("model", train_id)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    cont_filepath = "mobilenetv2-epoch_{epoch:02d}.hdf5"
    cont_checkpoint = ModelCheckpoint(os.path.join(save_dir, cont_filepath))

    best_filepath = "mobilenetv2-best.hdf5"
    best_checkpoint = ModelCheckpoint(os.path.join(save_dir, best_filepath), save_best_only=True, save_weights_only=True)

    plateau_scheduler = ReduceLROnPlateau(factor=0.2, patience=3, verbose=1, min_delta=0.005, min_lr=5e-7)

    history = model.fit(train_gen,
                        epochs=num_epochs,
                        steps_per_epoch=set_length['train'] // train_batch_size,
                        validation_data=val_gen,
                        validation_steps=validation_steps,
                        callbacks=[best_checkpoint, cont_checkpoint, plateau_scheduler],
                        class_weight=class_weight)

    saved_model_path = 'model/saved_model_dir'
    tf.saved_model.save(model, saved_model_path)