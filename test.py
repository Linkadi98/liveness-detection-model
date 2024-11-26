import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
import csv
from tqdm import tqdm

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--threshold", type=float)
parser.add_argument("-d", "--test_dir", type=str, default='test', help="path to 'test' folder")
parser.add_argument("-m", "--model_file", type=str, default='final_model.keras', help="path to '.keras' model file")
parser.add_argument("-f", "--output_file", type=str, default='result.csv', help="path to file with results")
parser.add_argument("-e", "--epoch", type=int, default=1)
parser.add_argument("-s", "--steps_per_epoch", type=int, default=1)

def predict_spoof(image_path, width, height, model, threshold):
    img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_resize = cv2.resize(img_array, (width, height))
    data = np.asarray(img_resize)
    to_predict = np.array([data], dtype=np.float32) / 255
    prediction_value = model.predict(to_predict, verbose=0)
    prediction_value = np.squeeze(prediction_value)

    return 0 if prediction_value < threshold else 1

if __name__ == '__main__':
    args = vars(parser.parse_args())

    test_dir = args["test_dir"]
    threshold = args["threshold"]
    model_file_name = args["model_file"]
    file_name = args["output_file"]
    epoch = args["epoch"]
    steps_per_epoch = args["steps_per_epoch"]

    if os.path.isfile(model_file_name):
        model = tf.keras.models.load_model(model_file_name)
        img_width, img_height = model.input.shape[1], model.input.shape[2]
        spoof_dir = os.path.join(test_dir, "spoof")
        real_dir = os.path.join(test_dir, "real")

        tp, tn, fp, fn = 0, 0, 0, 0

        for img in tqdm(os.listdir(spoof_dir), desc="Processing spoof images"):
            img_name, ext = os.path.splitext(img)
            if ext in [".png", ".jpg", ".bmp"]:
                prediction = predict_spoof(os.path.join(spoof_dir, img), img_width, img_height, model, threshold)
                if prediction == 0:
                    fp += 1
                else:
                    tn += 1

        for img in tqdm(os.listdir(real_dir), desc="Processing real images"):
            img_name, ext = os.path.splitext(img)
            if ext in [".png", ".jpg", ".bmp"]:
                prediction = predict_spoof(os.path.join(real_dir, img), img_width, img_height, model, threshold)
                if prediction == 1:
                    fn += 1
                else:
                    tp += 1

        total = tp + tn + fp + fn
        FAR = int(fp / max(total, 1) * 100)
        FRR = int(fn / max(total, 1) * 100)
        ACC = (tp + tn) / total * 100
        APCER = fp / (tn + fp)
        BPCER = fn / (tp + fn)
        ACER = (APCER + BPCER) / 2 * 100
        print(f"False Accept Rate %: {FAR}")
        print(f"False Rejection Rate %: {FRR}")
        print(f"Accuracy %: {ACC}")
        print(f"ACER %: {ACER}")
        with open(file_name, mode="a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, steps_per_epoch, threshold, FAR, FRR, ACC, ACER])
    else:
        print("There is no model file")