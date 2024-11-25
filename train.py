import os
import tensorflow as tf
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.utils import Sequence
from model import get_model
import cv2
import numpy as np

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--img_width", type=int, default=224)
parser.add_argument("-ht", "--img_height", type=int, default=224)
parser.add_argument("-sd", "--seed_number", type=int, default=24)
parser.add_argument("-tbs", "--train_batch_size", type=int, default=32)
parser.add_argument("-vbs", "--val_batch_size", type=int, default=32)
parser.add_argument("-e", "--num_epochs", type=int, default=50)
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

class DataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, img_size, augmentations):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.augmentations = augmentations
        self.indices = np.arange(len(self.image_paths))

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.image_paths[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]

        images = []
        for img_path in batch_images:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)  # Ensure all images are resized to the same shape
            if self.augmentations:
                image = self.augmentations(image=image)['image']
                image = np.transpose(image, (1, 2, 0))
            images.append(np.array(image))  # Convert to NumPy array

        return np.array(images), np.array(batch_labels)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

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
    evaluation_dir = os.path.join(input_dir, "LCC_FASD_evaluation")

    # Albumentations augmentations
    train_augmentations = A.Compose([
        A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_augmentations = A.Compose([
        A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Load image paths and labels
    def load_data(data_dir):
        image_paths = []
        labels = []
        for label, class_name in enumerate(['real', 'spoof']):
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(label)
        return image_paths, labels

    train_image_paths, train_labels = load_data(training_dir)
    val_image_paths, val_labels = load_data(evaluation_dir)

    # Data generators
    train_gen = DataGenerator(train_image_paths, train_labels, train_batch_size, (img_width, img_height), train_augmentations)
    val_gen = DataGenerator(val_image_paths, val_labels, val_batch_size, (img_width, img_height), val_augmentations)

    # Build the model
    model = get_model(img_width, img_height)

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

    plateau_scheduler = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.005, min_lr=5e-7
    )

    early_stopping = EarlyStopping(
        monitor='val_acc',
        patience=5,
        restore_best_weights=True
    )

    csv_logger = CSVLogger(os.path.join(model_dir, 'training_log.csv'), append=True)

    # Training
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=num_epochs,
        callbacks=[best_checkpoint, cont_checkpoint, plateau_scheduler, early_stopping, csv_logger],
    )

    # Save final model
    saved_model_path = os.path.join(model_dir, "final_model.keras")
    tf.keras.models.save_model(model, saved_model_path)