import os
import cv2
import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--threshold", type=float)
parser.add_argument("-d", "--test_dir", type=str, default='test', help="path to 'test' folder")
parser.add_argument("-m", "--model_file", type=str, default='model.tflite', help="path to '.tflite' model file")


def predict_spoof(image_path, width, height, interpreter):
    img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_resize = cv2.resize(img_array, (width, height))
    data = np.asarray(img_resize)
    to_predict = np.array([data], dtype=np.float32) / 255
    interpreter.set_tensor(input_details[0]['index'], to_predict)
    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[0]['index'])
    prediction_value = np.squeeze(tflite_results)

    return 0 if prediction_value < threshold else 1


if __name__ == '__main__':
    args = vars(parser.parse_args())

    test_dir = args["test_dir"]
    threshold = args["threshold"]
    model_file_name = args["model_file"]

    if os.path.isfile(model_file_name):
        interpreter = tf.lite.Interpreter(model_path=model_file_name)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        img_width = input_details[0]['shape'][1]
        img_height = input_details[0]['shape'][2]
        spoof_dir = os.path.join(test_dir, "spoof")
        real_dir = os.path.join(test_dir, "real")

        tp, tn, fp, fn = 0, 0, 0, 0

        for img in os.listdir(spoof_dir):
            img_name, ext = os.path.splitext(img)
            if ext in [".png", ".jpg", ".bmp"]:
                prediction = predict_spoof(os.path.join(spoof_dir, img), img_width, img_height, interpreter)
                if prediction == 0:
                    fp += 1
                else:
                    tn += 1

        for img in os.listdir(real_dir):
            img_name, ext = os.path.splitext(img)
            if ext in [".png", ".jpg", ".bmp"]:
                prediction = predict_spoof(os.path.join(real_dir, img), img_width, img_height, interpreter)
                if prediction == 1:
                    fn += 1
                else:
                    tp += 1

        total = tp + tn + fp + fn
        FAR = int(fp / max(total, 1) * 100)
        FRR = int(fn / max(total, 1) * 100)
        accuracy = (tp + tn) / total * 100
        APCER = fp / (tn + fp)
        BPCER = fn / (tp + fn)
        ACER = (APCER + BPCER) / 2 * 100
        print(f"False Accept Rate %: {FAR}")
        print(f"False Rejection Rate %: {FRR}")
        print(f"Accuracy %: {accuracy}")
        print(f"ACER %: {ACER}")
    else:
        print("There is no model.tflite file")
