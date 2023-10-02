import tensorflow as tf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--saved_model_dir", type=str, default=None, help="path to saved model dir")
parser.add_argument("-f", "--tflite_file_path", type=str, default='model.tflite', help="path to 'tflite' file")

if __name__ == '__main__':
    args = vars(parser.parse_args())

    saved_model_dir = args["saved_model_dir"]
    tflite_file = args["tflite_file_path"]
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    tf.io.write_file(tflite_file, tflite_model)
