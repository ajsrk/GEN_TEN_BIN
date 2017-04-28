
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts image data to TFRecords file format with Example protos.
The image data set is expected to reside in JPEG files located in the
following directory structure.
  data_dir/label_0/image0.jpeg
  data_dir/label_0/image1.jpg
  ...
  data_dir/label_1/weird-image.jpeg
  data_dir/label_1/my-image.jpeg
  ...
where the sub-directory is the unique label associated with these images.
This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files
  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-01023-of-01024
and
  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128
where we have selected 1024 and 128 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:
  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always 'JPEG'
  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [0, num_labels] where 0 is unused and left as
    the background class.
  image/class/text: string specifying the human-readable version of the label
    e.g. 'dog'
If your data set involves bounding boxes, please look at build_imagenet_data.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import pandas as pd
import tensorflow as tf
# May not need this import
from PIL import Image

import image_class_map as cmap


tf.app.flags.DEFINE_string('train_directory', '/home/ajsrk1207/super_bowl_data/data_dir/',
                          'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/home/ajsrk1207/super_bowl_data/data_dir/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('train_label_file', '/home/ajsrk1207/super_bowl_data/stage1_labels.csv',
                           'Label file for training set')
tf.app.flags.DEFINE_string('validation_label_file', '/home/ajsrk1207/super_bowl_data/stage1_labels.csv',
                           'Label file for validation set')
tf.app.flags.DEFINE_string('output_directory', '/home/ajsrk1207/super_bowl_data/output_dir/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 6,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 6,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 6,
                            'Number of threads to preprocess the images.')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.
tf.app.flags.DEFINE_string('labels_file', '/home/ajsrk1207/super_bowl_data/labels.txt', 'Labels file')


FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width, depth):
  """Build an Example proto for an example.
  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  colorspace = 'Grayscale'
  channels = 1
  array_format = 'ndarray'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/depth': _int64_feature(depth),
      'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
      'image/format': _bytes_feature(tf.compat.as_bytes(array_format)),
      'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
      'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
  return example



def _pre_process_image(filename):
  """Pre-processes single file
    Args:
      filename: string, path to the file
    Returns:
      pre_image: ndarray, a numpy array representing image data
  """
  i = Image.open(filename)
  img_array = np.asarray(i)
  return img_array


def _process_image(filename):
  """Process a single image file.
  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Get a numpy version of pre-processed data
  pre_processed_image = np.load(filename)

  # Check that we have a 3-d array
  assert len(pre_processed_image.shape) == 3
  height = pre_processed_image.shape[0]
  width = pre_processed_image.shape[1]
  depth = pre_processed_image.shape[2] #Can be a 3-d image or a 2-d RGB image
  image_data = pre_processed_image.tostring()

  return image_data, height, width, depth


def _process_image_files_batch(thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.
  Args:
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      text = texts[i]

      try:
        image_buffer, height, width, depth = _process_image(filename)
      except Exception as e:
        print(e)
        print('SKIPPED: Unexpected eror while decoding %s.' % filename)
        continue

      example = _convert_to_example(filename, image_buffer, label,
                                    text, height, width, depth)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, num_shards):
  """Process and save list of images as TFRecord of Example protos.
  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  assert len(filenames) == len(texts)
  assert len(filenames) == len(labels)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()


  threads = []
  for thread_index in range(len(ranges)):
    args = (thread_index, ranges, name, filenames,
            texts, labels, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(data_dir, labels_file, label_map_path):
  """Build a list of all images files and labels in the data set.
  Args:
    data_dir: string, path to the root directory of images.
      Assumes that the image data set resides in JPEG files located in
      the following directory structure.
        data_dir/dog/another-image.JPEG
        data_dir/dog/my-image.jpg
      where 'dog' is the label associated with these images.
    labels_file: string, path to the labels file.
      The labels file is a csv file that lists the base name  
       , integer label and string name of the class of all the files
  Returns:
    filenames: list of strings; each string is a path to an image file.
    texts: list of strings; each string is the class, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth.
  """  
  unique_labels = [l.strip() for l in tf.gfile.FastGFile(
      labels_file, 'r').readlines()]

  filenames, labels = cmap.get_image_label_list(label_map_path, data_dir)
  texts = []

  for l in labels:
    texts.append(unique_labels[l])


  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  texts = [texts[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Found %d  files across %d labels inside %s.' %
        (len(filenames), len(unique_labels), data_dir))
  return filenames, texts, labels


def _process_dataset(name, directory, num_shards, labels_file, label_map_path):
  """Process a complete data set and save it as a TFRecord.
  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """
  filenames, texts, labels = _find_image_files(directory, labels_file, label_map_path)
  _process_image_files(name, filenames, texts, labels, num_shards)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)

  train_label_map, validation_label_map = cmap.prep_img_label_maps(FLAGS.train_label_file,FLAGS.validation_label_file)

  # Run it!
  _process_dataset('validation', FLAGS.validation_directory,
                   FLAGS.validation_shards, FLAGS.labels_file, validation_label_map)
  _process_dataset('train', FLAGS.train_directory,
                   FLAGS.train_shards, FLAGS.labels_file, train_label_map)


if __name__ == '__main__':
  tf.app.run()
