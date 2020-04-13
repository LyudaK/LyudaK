import argparse
import sys
import os.path
import os
import math
import datetime, time
import numpy as np
from scipy.spatial.distance import squareform
from sklearn import preprocessing, model_selection, metrics
from sklearn.metrics import pairwise_distances
from scipy import misc
import scipy.cluster.hierarchy as hac


import numpy as np

np.random.seed(123)  # for reproducibility

use_my_cnn = False

KERAS, MXNET, TF = 0, 1, 2
use_framework = MXNET

DATASET_PATH = '/Users/lyudakopeikina/Downloads/lfw'  # _faces'

import tensorflow as tf

img_extensions = ['.jpg', '.jpeg', '.png']


def is_image(path):
    _, file_extension = os.path.splitext(path)
    return file_extension.lower() in img_extensions


def get_files(db_dir):
    return [[d, os.path.join(d, f)] for d in next(os.walk(db_dir))[1] for f in next(os.walk(os.path.join(db_dir, d)))[2]
            if not f.startswith(".") and is_image(f)]


def load_graph(frozen_graph_filename, prefix=''):
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=prefix)
    return graph


class TensorFlowInference:
    def __init__(self, frozen_graph_filename, input_tensor, output_tensor, learning_phase_tensor=None, convert2BGR=True,
                 imageNetUtilsMean=True, additional_input_value=0):
        graph = load_graph(frozen_graph_filename, '')
        print([n.name for n in graph.as_graph_def().node if 'input' in n.name])

        graph_op_list = list(graph.get_operations())
        print([n.name for n in graph_op_list if 'keras_learning' in n.name])

        self.tf_sess = tf.Session(graph=graph)

        self.tf_input_image = graph.get_tensor_by_name(input_tensor)
        print('tf_input_image=', self.tf_input_image)
        self.tf_output_features = graph.get_tensor_by_name(output_tensor)
        print('tf_output_features=', self.tf_output_features)
        self.tf_learning_phase = graph.get_tensor_by_name(learning_phase_tensor) if learning_phase_tensor else None;
        print('tf_learning_phase=', self.tf_learning_phase)
        if self.tf_input_image.shape.dims is None:
            w = h = 160
        else:
            _, w, h, _ = self.tf_input_image.shape
        self.w, self.h = int(w), int(h)
        print('input w,h', self.w, self.h, ' output shape:', self.tf_output_features.shape)
        # for n in graph.as_graph_def().node:
        #    print(n.name, n.op)
        # sys.exit(0)

        self.convert2BGR = convert2BGR
        self.imageNetUtilsMean = imageNetUtilsMean
        self.additional_input_value = additional_input_value

    def preprocess_image(self, img_filepath, crop_center):
        if crop_center:
            orig_w, orig_h = 250, 250
            img = misc.imread(img_filepath, mode='RGB')
            img = misc.imresize(img, (orig_w, orig_h), interp='bilinear')
            w1, h1 = 128, 128
            dw = (orig_w - w1) // 2
            dh = (orig_h - h1) // 2
            box = (dw, dh, orig_w - dw, orig_h - dh)
            img = img[dh:-dh, dw:-dw]
        else:
            img = misc.imread(img_filepath, mode='RGB')

        x = misc.imresize(img, (self.w, self.h), interp='bilinear').astype(float)

        if self.convert2BGR:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
            # Zero-center by mean pixel
            if self.imageNetUtilsMean:  # imagenet.utils caffe
                x[..., 0] -= 103.939
                x[..., 1] -= 116.779
                x[..., 2] -= 123.68
            else:  # vggface-2
                x[..., 0] -= 91.4953
                x[..., 1] -= 103.8827
                x[..., 2] -= 131.0912
        else:
            # x=(x-127.5)/128.0
            x /= 127.5
            x -= 1.
            # x=x/128.0-1.0
        return x

    def extract_features(self, img_filepath, crop_center=False):
        x = self.preprocess_image(img_filepath, crop_center)
        x = np.expand_dims(x, axis=0)
        feed_dict = {self.tf_input_image: x}
        if self.tf_learning_phase is not None:
            feed_dict[self.tf_learning_phase] = self.additional_input_value
        preds = self.tf_sess.run(self.tf_output_features, feed_dict=feed_dict).reshape(-1)
        # preds = self.tf_sess.run(self.tf_output_features, feed_dict=feed_dict).mean(axis=(0,1,2)).reshape(-1)
        return preds

    def close_session(self):
        self.tf_sess.close()

def load_recognizer():

    recognizer=TensorFlowInference('/Users/lyudakopeikina/Documents/models/insightface.pb',input_tensor='img_inputs:0',output_tensor='resnet_v1_50/E_BN2/Identity:0',learning_phase_tensor='dropout_rate:0',convert2BGR=False,additional_input_value=0.9)
    #recognizer = TensorFlowInference('/Users/lyudakopeikina/Documents/models/20180402-114759.pb',
    #                                  input_tensor='input:0', output_tensor='embeddings:0',
    #                                  learning_phase_tensor='phase_train:0',
    #                                  convert2BGR=False)


    return recognizer
def get_BCubed_set(y_vals):
    dic={}
    for i,y in enumerate (y_vals):
        dic[i]=set([y])
    return dic
def BCubed_stat(y_true, y_pred, beta=1.0):
    cdict=get_BCubed_set(y_true)
    ldict=get_BCubed_set(y_pred)
    p=precision(cdict, ldict)
    r=recall(cdict, ldict)
    f=fscore(p, r, beta)
    return (p,r,f)


# B-cubed
def fscore(p_val, r_val, beta=1.0):
    """Computes the F_{beta}-score of given precision and recall values."""
    return (1.0 + beta ** 2) * (p_val * r_val / (beta ** 2 * p_val + r_val))


def mult_precision(el1, el2, cdict, ldict):
    """Computes the multiplicity precision for two elements."""
    return min(len(cdict[el1] & cdict[el2]), len(ldict[el1] & ldict[el2])) \
           / float(len(cdict[el1] & cdict[el2]))


def mult_recall(el1, el2, cdict, ldict):
    """Computes the multiplicity recall for two elements."""
    return min(len(cdict[el1] & cdict[el2]), len(ldict[el1] & ldict[el2])) \
           / float(len(ldict[el1] & ldict[el2]))


def precision(cdict, ldict):
    """Computes overall extended BCubed precision for the C and L dicts."""
    return np.mean([np.mean([mult_precision(el1, el2, cdict, ldict) \
                             for el2 in cdict if cdict[el1] & cdict[el2]]) for el1 in cdict])


def recall(cdict, ldict):
    """Computes overall extended BCubed recall for the C and L dicts."""
    return np.mean([np.mean([mult_recall(el1, el2, cdict, ldict) \
                             for el2 in cdict if ldict[el1] & ldict[el2]]) for el1 in cdict])


def get_features_file():
    #features_file = 'lfw_features_insightface.npz'
    features_file = '/Users/lyudakopeikina/Downloads/lfw/lfw_features_vgg16.npz'
    if not os.path.exists(features_file):
            print('np')
            recognizer = load_recognizer()


            crop_center = False
            PATH_TO_DATA = '/Users/lyudakopeikina/Downloads/lfw'
            dirs_and_files = np.array([[d, os.path.join(d, f)] for d in next(os.walk(PATH_TO_DATA))[1] for f in
                                       next(os.walk(os.path.join(PATH_TO_DATA, d)))[2] if is_image(f)])
            # dirs_and_files=np.array([[d,os.path.join(d,f)] for d in next(os.walk(db_dir))[1] if d!='1' and d!='2' for f in next(os.walk(os.path.join(db_dir,d)))[2] if is_image(f)])
            print('dirs_files', dirs_and_files)

            dirs = dirs_and_files[:, 0]
            files = dirs_and_files[:, 1]
            # print(files)
            print('opened')
            print(dirs)
            print(len(np.unique(dirs)))
            print('files', files)

            label_enc=preprocessing.LabelEncoder()
            label_enc.fit(dirs)
            y_true=label_enc.transform(dirs)
            print ('y=',y_true)
            start_time = time.time()
            X=np.array([recognizer.extract_features(os.path.join(PATH_TO_DATA,filepath)) for filepath in files])


            np.savez(features_file, x=X, y_true=y_true)
            print('--- %s seconds ---' % (time.time() - start_time))
            print('X_train.shape=', X.shape)
            print('X_train[0,5]=', X[:, 0:6])


    data = np.load(features_file)
    X = data['x']
    X_norm = preprocessing.normalize(X, norm='l2')
    y_true = data['y_true']
    # y_true=data['y']

    dist_matrix = pairwise_distances(X_norm)

    distanceThreshold = 0.9 #0.97
    clusteringMethod = 'single'
    condensed_dist_matrix = squareform(dist_matrix, checks=False)
    z = hac.linkage(condensed_dist_matrix, method=clusteringMethod)
    y_pred = hac.fcluster(z, distanceThreshold, 'distance')

    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(y_true, y_pred)
    bcubed_precision, bcubed_recall, bcubed_fmeasure = BCubed_stat(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    print('ari',ari)
    print('ami',ami)
    print('bcubed_precision',bcubed_precision)
    print('bcubed_recall', bcubed_recall)
    print('bcubed_f', bcubed_fmeasure)
    print('Galagher dataset: # classes=%d #clusters=%d' % (len(np.unique(y_true)), len(np.unique(y_pred))))
    print('homogeneity=%0.3f, completeness=%0.3f' % (homogeneity, completeness))

if __name__ == '__main__':
    get_features_file()


