import pytesseract
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
from rembg import remove

from paddleocr import PaddleOCR, draw_ocr
import re

pytesseract.pytesseract.tesseract_cmd ="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


from pathlib import Path
from collections import Counter

import tensorflow as tf
import keras 
from keras import layers

from keras import ops
from keras.models import load_model
from keras.utils import custom_object_scope

from keras import ops
os.environ["KERAS_BACKEND"] = "tensorflow"



def replace_characters(s):

    replacements = {'Æ': 'AE',
                'æ': 'ae',
                'Å': 'A',
                'å': 'a',
                'Ø': 'O',
                'ø': 'o'}
    
    for old_char, new_char in replacements.items():
        s = s.replace(old_char, new_char)
    return s


def detect_text(img, a=31, b=5, apply_threshold = True):

    #img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if apply_threshold == True:

        adaptive_threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, a, b)
    else:
        adaptive_threshold = img

    results = pytesseract.image_to_data(adaptive_threshold, lang='dan')
    n_detections = 0
    for id, line in enumerate(results.splitlines()):

        if id != 0:
            line = line.split()

            if len(line) == 12 and float(line[10]) > 80.0:
                n_detections = n_detections+1

    return n_detections


def black_pixel_percentage(img):

    img_size = img.size
    black_pixels = np.sum(img == 0)

    return (black_pixels / img_size) * 100

#__________________________________________________________________
#___________________________________________________________________

def white_pixel_percentage(img):

    img_size = img.size
    white_pixels = np.sum(img == 255)

    return (white_pixels / img_size) * 100

#_______________________________________________________________________
#_______________________________________________________________________

def remove_white_borders(image):

    _, binary = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)
    coords = cv2.findNonZero(binary)

    if coords is None: 
        return image

    x, y, w, h = cv2.boundingRect(coords)

    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

#_____________________________________________________________________
#_____________________________________________________________________

def remove_black_borders(image):

    _, binary = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)

    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

#__________________________________________________________________
#__________________________________________________________________

def vertical_perspective_transform(image, value, padding):

    padded_image = cv2.copyMakeBorder(image, padding, 0, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    rows, cols = padded_image.shape[:2]
    src_points = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

    dst_points = np.float32([[-1*value, -1*value], [cols+value, -1*value], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_image = cv2.warpPerspective(padded_image, M, (cols, rows))

    # Save the transformed image
    return transformed_image

#___________________________________________________________________
#_____________________________________________________________________

def horizontal_perspective_transform(image, value, padding, side):

    if side == "left":

        padded_image = cv2.copyMakeBorder(image, padding, padding, padding, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        rows, cols = padded_image.shape[:2]
        src_points = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

        dst_points = np.float32([[-1*value, -1*value], [cols, 0], [-1*value, rows+value], [cols, rows]])
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed_image = cv2.warpPerspective(padded_image, M, (cols, rows))

    if side == "right":

        padded_image = cv2.copyMakeBorder(image, padding, padding, padding, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        rows, cols = padded_image.shape[:2]
        src_points = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

        dst_points = np.float32([[0, 0], [cols+value, value], [0, rows], [cols+value, rows+value]])
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed_image = cv2.warpPerspective(padded_image, M, (cols, rows))


    # Save the transformed image
    return transformed_image

def apply_best_vertical_transformation(img):

    values = [50, 100, 200, 300, 400, 500]
    n_detections_prev = 0
    vr_transform_prev = img

    for value in values:
        if value > 100:
            padding = value - 100
        else:
            padding = 150
        vr_transform = vertical_perspective_transform(img, value, padding)
        vr_transform = remove_black_borders(vr_transform)
        n_detections = detect_text(vr_transform, 51, 25, apply_threshold=True)
        print(n_detections)
        if not n_detections > n_detections_prev:
            break
        else:
            n_detections_prev = n_detections
            vr_transform_prev = vr_transform

    
    return vr_transform_prev

def apply_best_horizontal_transformation_left(img):

    values = [50, 100, 200, 300, 400, 500]
    n_detections_prev = 0
    hr_transform_prev = img

    for value in values:
        if value > 100:
            padding = value - 100
        else:
            padding = 150
        hr_transform = horizontal_perspective_transform(img, value, padding, side="left")
        hr_transform = remove_black_borders(hr_transform)
        n_detections = detect_text(hr_transform, 51, 25, apply_threshold=True)
        print(n_detections)
        if not n_detections > n_detections_prev:
            break
        else:
            n_detections_prev = n_detections
            hr_transform_prev = hr_transform

    
    return hr_transform_prev

def apply_best_horizontal_transformation_right(img):

    values = [50, 100, 200, 300, 400, 500]
    n_detections_prev = 0
    hr_transform_prev = img

    for value in values:
        if value > 100:
            padding = value - 100
        else:
            padding = 150
        hr_transform = horizontal_perspective_transform(img, value, padding, side="right")
        hr_transform = remove_black_borders(hr_transform)
        n_detections = detect_text(hr_transform, 51, 25, apply_threshold= True)
        print(n_detections)
        if not n_detections > n_detections_prev:
            break
        else:
            n_detections_prev = n_detections
            hr_transform_prev = hr_transform

    
    return hr_transform_prev

def apply_adaptive_threshold(img):
    best_values = []
    values = [[15, 4],
              [23, 7],
              [29, 12],
              [35, 15],
              [41, 18],
              [51, 25]]
    for value in values:

        adaptive_threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, value[0], value[1])
        n_detection = detect_text(adaptive_threshold, apply_threshold= False)
        best_values.append(n_detection)

    index = best_values.index(max(best_values))

    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, values[index][0], values[index][1])

def distortion_free_resize(image, img_size):
  w, h = img_size
  image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

  # Check tha amount of padding needed to be done.
  pad_height = h - tf.shape(image)[0]
  pad_width = w - tf.shape(image)[1]

  # only necessary if you want to do same amount of padding on both sides.
  if pad_height % 2 != 0:
    height = pad_height // 2
    pad_height_top = height +1
    pad_height_bottom = height
  else:
    pad_height_top = pad_height_bottom = pad_height // 2

  if pad_width % 2 != 0:
    width = pad_width // 2
    pad_width_left = width + 1
    pad_width_right = width
  else:
    pad_width_left = pad_width_right = pad_width // 2

  image = tf.pad(
      image, paddings=[
          [pad_height_top, pad_height_bottom],
          [pad_width_left, pad_width_right],
          [0, 0],
      ],
      constant_values=255.0
  )
  #image = tf.transpose(image, perm=[1,0,2])
  #image = tf.image.flip_left_right(image)
  return image

batch_size = 16
padding_token = 99
image_width = 200
image_height = 50

def preprocess_image(image_path, img_size=(image_width, image_height)):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_png(image, 1)
  #image = distortion_free_resize(image, img_size)
  image = tf.cast(image, tf.float32) / 255.0
  return image

def process_images_labels(image_path, label):
  image = preprocess_image(image_path)
  return {"image": image, "label": "i"}
  
def prepare_dataset(image_paths, labels):
  dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
    process_images_labels, num_parallel_calls=AUTOTUNE
  )

  return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)

def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    label_length = ops.cast(ops.squeeze(label_length, axis=-1), dtype="int32")
    input_length = ops.cast(ops.squeeze(input_length, axis=-1), dtype="int32")
    sparse_labels = ops.cast(
        ctc_label_dense_to_sparse(y_true, label_length), dtype="int32"
    )

    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())

    return ops.expand_dims(
        tf.compat.v1.nn.ctc_loss(
            inputs=y_pred, labels=sparse_labels, sequence_length=input_length
        ),
        1,
    )


def ctc_label_dense_to_sparse(labels, label_lengths):
    label_shape = ops.shape(labels)
    num_batches_tns = ops.stack([label_shape[0]])
    max_num_labels_tns = ops.stack([label_shape[1]])

    def range_less_than(old_input, current_input):
        return ops.expand_dims(ops.arange(ops.shape(old_input)[1]), 0) < tf.fill(
            max_num_labels_tns, current_input
        )

    init = ops.cast(tf.fill([1, label_shape[1]], 0), dtype="bool")
    dense_mask = tf.compat.v1.scan(
        range_less_than, label_lengths, initializer=init, parallel_iterations=1
    )
    dense_mask = dense_mask[:, 0, :]

    label_array = ops.reshape(
        ops.tile(ops.arange(0, label_shape[1]), num_batches_tns), label_shape
    )
    label_ind = tf.compat.v1.boolean_mask(label_array, dense_mask)

    batch_array = ops.transpose(
        ops.reshape(
            ops.tile(ops.arange(0, label_shape[0]), max_num_labels_tns),
            tf.reverse(label_shape, [0]),
        )
    )
    batch_ind = tf.compat.v1.boolean_mask(batch_array, dense_mask)
    indices = ops.transpose(
        ops.reshape(ops.concatenate([batch_ind, label_ind], axis=0), [2, -1])
    )

    vals_sparse = tf.compat.v1.gather_nd(labels, indices)

    return tf.SparseTensor(
        ops.cast(indices, dtype="int64"), 
        vals_sparse, 
        ops.cast(label_shape, dtype="int64")
    )

class CTCLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
        input_length = ops.cast(ops.shape(y_pred)[1], dtype="int64")
        label_length = ops.cast(ops.shape(y_true)[1], dtype="int64")

        input_length = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * ops.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    input_shape = ops.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())
    input_length = ops.cast(input_length, dtype="int32")

    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length
        )
    else:
        (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
        )
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    return (decoded_dense, log_prob)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

characters = ['!', '"', '&', "'", ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '8', '9', ':', ';', '=', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', '[', '\\', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '¢', '£', '©', '»', 'é', '—', '‘', '€', '™', 'ﬂ']

AUTOTUNE = tf.data.AUTOTUNE

# Maping characaters to integers
char_to_num = layers.StringLookup(vocabulary=characters, mask_token=None)

#Maping integers back to original characters
num_to_chars = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)


path = "test.png"
img = cv2.imread(path, 0)

keyword = "UDDANNELSESBEVIS".lower()

keyword_for_paddle_ocr = replace_characters(keyword)

#remove white borders and check for digital or scanned
white_removed_img = remove_white_borders(img)
white_pixel_percent = white_pixel_percentage(white_removed_img)

if white_pixel_percent > 30.0:
    extracted_text = pytesseract.image_to_string(img, lang='dan')

    match = re.search(re.escape(keyword), extracted_text.lower())
    if match:
        print(f"'{keyword}' found")
    else:
        print(f"'{keyword}' not found")

else:
    #it is a scanned image
    #first remove background

    image = remove(white_removed_img)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    #pre_process_for_removed_background
    #apply_vertical_transformation
    
    image = apply_best_vertical_transformation(image)
    image = apply_best_horizontal_transformation_left(image)
    image = apply_best_horizontal_transformation_right(image)

    blurred = cv2.GaussianBlur(image, (5,5), 2)
    deblurred_image = cv2.addWeighted(image, 6.5, blurred, -5.5, 0)


    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(deblurred_image, cls=True)
    texts = [line[1][0] for line in result[0]]

    joined_string = " ".join(texts)
    match = re.search(re.escape(keyword_for_paddle_ocr), joined_string.lower())

    if match:
        print(f"'{keyword}' found")

    else:
        deblurred_image = apply_adaptive_threshold(deblurred_image)

        cropped_images = []

        results = pytesseract.image_to_data(deblurred_image, lang="dan")

        for id, line in enumerate(results.splitlines()):

            if id != 0:
                line = line.split()

                if len(line) == 12:
                    x, y, w, h = int(line[6]), int(line[7]), int(line[8]), int(line[9])
                    cropped_image = deblurred_image[y:y+h, x:x+w]
                    cropped_image = distortion_free_resize(cropped_image, (200, 50))
                    cropped_images.append(cropped_image)

        y_valid = []
        validation_dataset = prepare_dataset(cropped_images, y_valid)

        with custom_object_scope({'CTCLayer': CTCLayer}):
            model = load_model('ocr_model.h5')

        prediction_model = keras.models.Model(
            model.input[0], model.get_layer(name="dense2").output)
        
        extracted = []

        for batch in validation_dataset.take(10):
            batch_images = batch["image"]
            batch_labels = batch["label"]

            preds = prediction_model.predict(batch_images)
            pred_texts = decode_batch_predictions(preds)

            for i in range(len(pred_texts)):
                text = pred_texts[i].split("[UNK]")[0]
                extracted.append(text)

        joined_string = " ".join(extracted)
        match = re.search(re.escape(keyword), joined_string.lower())

        if match:
            print(f"'{keyword}' found")