import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_file", type=str, required=True, help="path to '.h5' or '.keras' model file")
parser.add_argument("-f", "--tflite_file_path", type=str, default='model.tflite', help="path to 'tflite' file")

if __name__ == '__main__':
    args = vars(parser.parse_args())

    model_file = args["model_file"]
    tflite_file = args["tflite_file_path"]

    # Load the Keras model
    model = tf.keras.models.load_model(model_file)

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the converted model to a file
    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)