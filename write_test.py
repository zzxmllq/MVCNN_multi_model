import numpy as np
import os
import cv2
import tensorflow as tf

W = H = 256
filename_train = "/media/zzxmllq/0002605F0001462E/mvcnn/test_224.tfrecords"
writer = tf.python_io.TFRecordWriter(filename_train)
path = "/media/zzxmllq/0002605F0001462E/mvcnn/modelnet40v1/"
dir_class = os.listdir(path)
for index, _class in enumerate(dir_class):
    dir_test_train = os.listdir(path + _class)
    for test_train in dir_test_train:
        if test_train == "test":
            dir_data = os.listdir(path + _class + "/" + test_train)
            dir_data.sort()
            view_data_list = []
            i = 0
            num12 = 0
            for data in dir_data:
                img = cv2.imread(path + _class + "/" + test_train + "/" + data)
                # img = cv2.resize(img, (W, H))
                img = cv2.bitwise_not(img)
                img = img.astype('float32')
                # left = 256 / 2 - 227 / 2
                # top = 256 / 2 - 227 / 2
                # right = left + 227
                # bottom = top + 227
                # img = img[left:right, top:bottom, :]
                # print(len(dir_data))

                # print img.type
                view_data_list.append(img)

                if (i + 1) % 12 == 0:
                    num12 += 1
                    img_views = np.array(view_data_list)
                    # print img_views.shape
                    img_views_raw = img_views.tostring()
                    # img_raw = img.tostring()
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_views_raw])),
                                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index]))
                            }
                        )
                    )
                    serialized = example.SerializeToString()
                    writer.write(serialized)
                    # i = 0
                    view_data_list = []
                    # print index
                # print(index)
                i = i + 1
            print(num12)
writer.close()
