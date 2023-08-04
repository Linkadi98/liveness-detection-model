import os
import cv2
import numpy as np
import tensorflow as tf

model_file_name = "model/model.tflite"
test_dir = "test"
threshold = 0.5


def predict_spoof(image_path, width, height, interpreter):
    img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_resize = cv2.resize(img_array, (width, height))
    data = np.asarray(img_resize)
    to_predict = np.array([data], dtype=np.float32) / 255
    interpreter.set_tensor(input_details[0]['index'], to_predict)
    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[0]['index'])
    prediction_value = np.squeeze(tflite_results)
    prediction = np.zeros(prediction_value.shape).astype(np.int32)  # Sigmoid
    prediction[prediction_value > threshold] = 1

    return prediction


if __name__ == '__main__':
    if os.path.isfile(model_file_name):
        interpreter = tf.lite.Interpreter(model_path=model_file_name)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        img_width = input_details[0]['shape'][1]
        img_height = input_details[0]['shape'][2]
        spoof_dir = os.path.join(test_dir, "spoof")
        real_dir = os.path.join(test_dir, "real")

        false_accepts = 0
        false_rejects = 0
        total = 0

        for img in os.listdir(spoof_dir):
            img_name, ext = os.path.splitext(img)
            if ext in [".png", ".jpg", ".bmp"]:
                prediction = predict_spoof(os.path.join(spoof_dir, img), img_width, img_height, interpreter)
                if prediction != 1:
                    false_accepts += 1
                total += 1

        for img in os.listdir(real_dir):
            img_name, ext = os.path.splitext(img)
            if ext in [".png", ".jpg", ".bmp"]:
                prediction = predict_spoof(os.path.join(real_dir, img), img_width, img_height, interpreter)
                if prediction == 1:
                    false_rejects += 1
                total += 1

        FAR = int(false_accepts / max(total, 1) * 100)
        FRR = int(false_rejects / max(total, 1) * 100)
        print(f"False Accept Rate %: {FAR}")
        print(f"False Rejection Rate %: {FRR}")
    else:
        print("There is no model.tflite file")
