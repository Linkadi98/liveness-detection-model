import tensorflow as tf


if __name__ == '__main__':
    converter = tf.lite.TFLiteConverter.from_saved_model('model/saved_model_dir')
    tflite_model = converter.convert()
    tflite_model_path = 'model/model.tflite'
    tf.io.write_file(tflite_model_path, tflite_model)