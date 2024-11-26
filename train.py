import os
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from keras.src.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.src.preprocessing.image import ImageDataGenerator

from model import GetModel

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--img_width", type=int, default=224)
parser.add_argument("-ht", "--img_height", type=int, default=224)
parser.add_argument("-sd", "--seed_number", type=int, default=24)
parser.add_argument("-tbs", "--train_batch_size", type=int, default=32)
parser.add_argument("-vbs", "--val_batch_size", type=int, default=32)
parser.add_argument("-e", "--num_epochs", type=int, default=10)
parser.add_argument("-i", "--input_data_path", type=str, required=True, help="Path to resized dataset root folder")
parser.add_argument("-m", "--model_path", type=str, default='model', help="Path to model output folder")

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    args = vars(parser.parse_args())

    # Parameters
    img_width = args["img_width"]
    img_height = args["img_height"]
    seed_number = args["seed_number"]
    train_batch_size = args["train_batch_size"]
    val_batch_size = args["val_batch_size"]
    num_epochs = args["num_epochs"]
    input_dir = args["input_data_path"]
    model_dir = args["model_path"]

    # Define dataset paths
    training_dir = os.path.join(input_dir, "LCC_FASD_training")
    development_dir = os.path.join(input_dir, "LCC_FASD_development")
    evaluation_dir = os.path.join(input_dir, "LCC_FASD_evaluation")

    # ImageDataGenerators
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Data generators
    train_gen = train_datagen.flow_from_directory(
        training_dir,
        target_size=(img_width, img_height),
        batch_size=train_batch_size,
        class_mode="binary",
        seed=seed_number,
    )
    val_gen = val_datagen.flow_from_directory(
        evaluation_dir,
        target_size=(img_width, img_height),
        batch_size=val_batch_size,
        class_mode="binary",
        seed=seed_number,
    )

    # Build the model
    model = GetModel(img_width, img_height)

    # Callbacks
    save_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    best_checkpoint = ModelCheckpoint(
        filepath=os.path.join(save_dir, "mobilenetv3-best.weights.h5"),
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    cont_checkpoint = ModelCheckpoint(
        filepath=os.path.join(save_dir, "mobilenetv3-epoch_{epoch:02d}.weights.h5"),
        save_weights_only=True,
        verbose=1,
    )

    plateau_scheduler = ReduceLROnPlateau(factor=0.1, patience=2, verbose=1, min_delta=0.005, min_lr=1e-7)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Training
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=num_epochs,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        callbacks=[best_checkpoint, cont_checkpoint, plateau_scheduler, early_stopping],
    )

    # Save final model
    saved_model_path = os.path.join(model_dir, "final_model.h5")
    tf.keras.models.save_model(model, saved_model_path)

    # Plot metrics
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["acc"], label="Train Accuracy")
    plt.plot(history.history["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Save metrics plot
    metrics_plot_path = os.path.join(model_dir, "training_metrics.png")
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches="tight")
