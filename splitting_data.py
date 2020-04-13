from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time, datetime
import cv2
import shutil
import pickle

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report,accuracy_score, f1_score

import hac as hac
import numpy as np
import tensorflow as tf
import scipy.cluster.hierarchy as hac
from scipy import misc
from scipy.spatial.distance import squareform

from configparser import ConfigParser

from facial_analysis import FacialImageProcessing, get_video_file_orientation, is_image, is_video
#from facial_clustering import get_facial_clusters

# config values
from sklearn import preprocessing

minDaysDifferenceBetweenPhotoMDates = 2
minNoPhotos = 2
minNoFrames = 10
distanceThreshold = 0.95
minFaceWidthPercent = 0.05

img_size = 224
def load_graph(frozen_graph_filename, prefix=''):
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=prefix)
    return graph
def get_facial_clusters(dist_matrix, distanceThreshold=1, all_indices=None, no_images_in_cluster=1):
    clusters = []
    num_faces = dist_matrix.shape[0]
    clusteringMethod='weighted'
    #clusteringMethod = 'average'
    condensed_dist_matrix = squareform(dist_matrix, checks=False)
    z = hac.linkage(condensed_dist_matrix, method=clusteringMethod)
    labels = hac.fcluster(z, distanceThreshold, 'distance')

    if all_indices is None:
        clusters = [[ind for ind, label in enumerate(labels) if label == lbl] for lbl in set(labels)]
    else:
        for lbl in set(labels):
            cluster = [ind for ind, label in enumerate(labels) if label == lbl]
            if len(cluster) > 1:
                inf_dist = 100
                dist_matrix_cluster = dist_matrix[cluster][:, cluster]
                penalties = np.array(
                    [[inf_dist * (all_indices[i] == all_indices[j] and i != j) for j in cluster] for i in cluster])
                dist_matrix_cluster += penalties
                condensed_dist_matrix = squareform(dist_matrix_cluster)
                z = hac.linkage(condensed_dist_matrix, method=clusteringMethod)
                labels_cluster = hac.fcluster(z, inf_dist / 2, 'distance')
                clusters.extend([[cluster[ind] for ind, label in enumerate(labels_cluster) if label == l] for l in
                                    set(labels_cluster)])
            else:
                clusters.append(cluster)
    clusters.sort(key=len, reverse=True)
    return clusters

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


class FeatureExtractor:
    def __init__(self, vggmodel=None):
        if vggmodel is None:
            print('notvgg')
            self.tfInference = TensorFlowInference('/Users/lyudakopeikina/Documents/models/insightface.pb',
                                                   input_tensor='img_inputs:0',
                                                   output_tensor='resnet_v1_50/E_BN2/Identity:0',
                                                   learning_phase_tensor='dropout_rate:0', convert2BGR=False,
                                                   additional_input_value=0.9)
            #self.tfInference = TensorFlowInference('age_gender_tf2_new-01-0.14-0.92.pb', input_tensor='input_1:0',
                                                output_tensor='global_pooling/Mean:0')
            #self.tfInference = TensorFlowInference('D:/models//20180402-114759.pb', input_tensor='input:0', output_tensor='embeddings:0',
              #                               learning_phase_tensor='phase_train:0',
               #                              convert2BGR=False)
        else:
            #print('vgg')
            self.tfInference = None

            from keras_vggface.vggface import VGGFace
            from keras.engine import Model
            layers = {'vgg16': 'fc7/relu', 'resnet50': 'avg_pool'}
            model = VGGFace(model=vggmodel)
            out = model.get_layer(layers[vggmodel]).output
            self.cnn_model = Model(model.input, out)
            _, w, h, _ = model.input.shape
            self.size = (int(w), int(h))

    def extract_features(self, image_path):
        if self.tfInference is not None:
            #print('notVGGextract')
            return self.tfInference.extract_features(image_path)

        else:
            #print('VGGextract')
            from keras_vggface.utils import preprocess_input
            from keras.preprocessing import image
            img = image.load_img(image_path, target_size=self.size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = self.cnn_model.predict(x).reshape(-1)
            #print('preds',len(preds))
            return preds

    def close(self):
        if self.tfInference is not None:
            self.tfInference.close_session()

featureExtractor=None
def process_image(imgProcessing, img):


    def process(img):
        global featureExtractor
        if featureExtractor is None:
            featureExtractor = FeatureExtractor(model_desc[model_ind][0])
            #print('uahere')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = time.time()
        bounding_boxes, points = imgProcessing.detect_faces(img)
        elapsed = time.time() - t
        facial_features, bboxes = [], []
        for b in bounding_boxes:
            b = [int(bi) for bi in b]
            # print(b,img.shape)
            x1, y1, x2, y2 = b[0:4]
            if x2 > x1 and y2 > y1:
                img_h, img_w, _ = img.shape
                w, h = x2 - x1, y2 - y1
                dw, dh = 10, 10  # max(w//8,10),max(h//8,10) #w//6,h//6
                # sz=max(w+2*dw,h+2*dh)
                # dw,dh=(sz-w)//2,(sz-h)//2
                x1, x2 = x1 - dw, x2 + dw
                y1, y2 = y1 - dh, y2 + dh

                boxes = [[x1, y1, x2, y2]]

                if False:  # oversampling
                    delta = 10
                    boxes.append([x1 - delta, y1 - delta, x2 - delta, y2 - delta])
                    boxes.append([x1 - delta, y1 + delta, x2 - delta, y2 + delta])
                    boxes.append([x1 + delta, y1 - delta, x2 + delta, y2 - delta])
                    boxes.append([x1 + delta, y1 + delta, x2 + delta, y2 + delta])

                for ind in range(len(boxes)):
                    if boxes[ind][0] < 0:
                        boxes[ind][0] = 0
                    if boxes[ind][2] > img_w:
                        boxes[ind][2] = img_w
                    if boxes[ind][1] < 0:
                        boxes[ind][1] = 0
                    if boxes[ind][3] > img_h:
                        boxes[ind][3] = img_h

                for (x1, y1, x2, y2) in boxes[::-1]:
                    face_img = img[y1:y2, x1:x2, :]
                    cv2.imwrite('face.jpg', cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                    face_img = 'face.jpg'
                    t = time.time()
                    features = featureExtractor.extract_features(face_img)

                facial_features.append(features)
                bboxes.append(boxes[0])
        #print('facial_features', facial_features)
        return bboxes, points, facial_features
    #print(model_desc[model_ind][0])

    height, width, channels = img.shape
    bounding_boxes, _, facial_features = process(img)
    facial_images = []
    has_center_face = False
    for bb in bounding_boxes:
        x1, y1, x2, y2 = bb[0:4]
        face_img = cv2.resize(img[y1:y2, x1:x2, :], (img_size, img_size))
        facial_images.append(face_img)
        # dx=1.5*(x2-x1)
        #if (x2 - x1) / width >= minFaceWidthPercent:  # x1-dx<=width/2<=x2+dx:
          #  has_center_face = True
    #print('facial_images',len(facial_images))
    #print('has_center_face',has_center_face)
    #('ff',len(facial_features[0]))
    return facial_images, facial_features, has_center_face
def perform_splitting(all_indices, all_features, no_images_in_cluster):
    def feature_distance(i, j):
        dist = np.sqrt(np.sum((all_features[i] - all_features[j]) ** 2))
        #max_year = max(mdates[all_indices[i]].tm_year, mdates[all_indices[j]].tm_year)
        #cur_age_i, cur_age_j = max_year - all_born_years[i], max_year - all_born_years[j]
        #age_dist = (cur_age_i - cur_age_j) ** 2 / (cur_age_i + cur_age_j)
        #return [dist, age_dist * 0.1]
        return [dist]

    num_faces = len(all_indices)
    #print('num_faces',num_faces)
    if num_faces < no_images_in_cluster:
        return []

    t = time.time()
    pair_dist = np.array([[feature_distance(i, j) for j in range(num_faces)] for i in range(num_faces)])
    dist_matrix = np.clip(np.sum(pair_dist, axis=2), a_min=0, a_max=None)
    print('all_indices',all_indices)
    clusters = get_facial_clusters(dist_matrix, distanceThreshold, all_indices, no_images_in_cluster)
    elapsed = time.time() - t
    # print('clustering elapsed=%f'%(elapsed))

    print('clusters', clusters)
    print('len',len(clusters))

    def is_good_cluster(cluster):
        #print(len(cluster))
        res = len(cluster) >= no_images_in_cluster
        #print(res)
        return res
    def private_indices(all_indices,filtered_clusters):
        file_face = []
        c = 0
        private_photos=[]
        private_faces=[]
        for file in (all_indices):
            file_face.append([file, c])
            c = c + 1
        #print('file_face', file_face)
        for ff in file_face:
            for fc in filtered_clusters:
                if(ff[1] in fc):
                    private_photos.append(ff[0])
        return private_photos,file_face

    filtered_clusters = [cluster for cluster in clusters if is_good_cluster(cluster)]
    print('filtered_clusters',filtered_clusters)
    print('len',len(filtered_clusters))
    private_photos,file_face=private_indices(all_indices,filtered_clusters)
    print('private_photos',private_photos)

    return set(private_photos),file_face,clusters

def process_album(imgProcessing, album_dir):
    features_file = os.path.join(album_dir, 'features%s.dump'%model_desc[model_ind][1])
    t = time.time()
    if os.path.exists(features_file):
        with open(features_file, "rb") as f:
            files = pickle.load(f)
            all_facial_images = pickle.load(f)
            all_features = pickle.load(f)
            all_indices = pickle.load(f)
            private_photo_indices = pickle.load(f)
            y_true=pickle.load(f)
    else:
        # process static images
        dirs_and_files = np.array([[d, os.path.join(d, f)] for d in next(os.walk(album_dir))[1] for f
                                   in next(os.walk(os.path.join(album_dir, d)))[2] if is_image(f)])
        # dirs_and_files=np.array([[d,os.path.join(d,f)] for d in next(os.walk(db_dir))[1] if d!='1' and d!='2' for f in next(os.walk(os.path.join(db_dir,d)))[2] if is_image(f)])
        #print('dirs_files', dirs_and_files)

        dirs = dirs_and_files[:, 0]
        files = dirs_and_files[:, 1]
        # print(files)
        # print('opened')
        print(dirs)
        print(len(np.unique(dirs)))
        print('files',files)

        label_enc = preprocessing.LabelEncoder()
        label_enc.fit(dirs)
        y_true = label_enc.transform(dirs)
        print('y=', y_true)
        #files = [f for f in next(os.walk(album_dir))[2] if is_image(f)]
        # files=files[:20]
        all_facial_images, all_features, all_indices, private_photo_indices = [], [], [], []
        for i, fpath in enumerate(files):
            print(fpath)
            full_photo = cv2.imread(os.path.join(album_dir, fpath))
            facial_images, facial_features, has_center_face = process_image(imgProcessing, full_photo)
            if len(facial_images) == 0:
                full_photo_t = cv2.transpose(full_photo)
                rotate90 = cv2.flip(full_photo_t, 1)
                facial_images, facial_features, has_center_face = process_image(imgProcessing, rotate90)
                if len(facial_images) == 0:
                    rotate270 = cv2.flip(full_photo_t, 0)
                    facial_images, facial_features, has_center_face = process_image(imgProcessing,
                                                                                                   rotate270)
            if has_center_face:
                #private_photo_indices.append(i)
                print('hello ind')
            all_facial_images.extend(facial_images)
            for features in facial_features:
                features = features / np.sqrt(np.sum(features ** 2))
                all_features.append(features)
            all_indices.extend([i]*len(facial_images))
            #print('all_features',all_features)


            print('Processed photos: %d/%d\r' % (i + 1, len(files)), end='')
            sys.stdout.flush()

        with open(features_file, "wb") as f:
            pickle.dump(files, f)
            pickle.dump(all_facial_images, f)
            pickle.dump(all_features, f)
            pickle.dump(all_indices, f)
            pickle.dump(private_photo_indices, f)
            pickle.dump(y_true,f)
        print('features dumped into', features_file)

    elapsed = time.time() - t
    no_image_files = len(files)
    print('\nelapsed=%f for processing of %d files' % (elapsed, no_image_files))



    all_features = np.array(all_features)
    print(np.shape(all_features))

    private_photo_indices,file_face,all_clusters = perform_splitting(all_indices, all_features, minNoPhotos)
    print(private_photo_indices)
    #files = [f for f in next(os.walk(album_dir))[2] if is_image(f)]
    print('len(files)',len(files))
    y_pred = np.ones(len(files))
    for i in private_photo_indices:
        y_pred[i]=0
    print('y_true', y_true)
    print('y_pred', y_pred)
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('F1 score:', f1_score(y_true, y_pred))
    print('Recall:', recall_score(y_true, y_pred))
    print('Precision:', precision_score(y_true, y_pred))
    print('\n clasification report:\n', classification_report(y_true, y_pred))
    print('\n confussion matrix:\n', confusion_matrix(y_true, y_pred))
    false_negative=[i for i in range(len(files)//2) if i not in private_photo_indices]
    print(false_negative)
    print(file_face)
    fn_faces=[ff[1] for ff in file_face if ff[0] in false_negative]
    print(fn_faces)


    if True:
        res_dir = os.path.join(album_dir, 'clusters')
        if os.path.exists(res_dir):
            shutil.rmtree(res_dir, ignore_errors=True)
            time.sleep(2)

        clust_dir = os.path.join(res_dir, 'private')
        os.makedirs(clust_dir)
        for ind in private_photo_indices:
            full_photo = cv2.imread(os.path.join(album_dir, files[ind]))
            r = 200.0 / full_photo.shape[1]
            dim = (200, int(full_photo.shape[0] * r))
            full_photo = cv2.resize(full_photo, dim)
            cv2.imwrite(os.path.join(clust_dir, '%s.jpg' % (files[ind].split("\\")[1])), full_photo)
        clust_dir = os.path.join(res_dir, 'public')
        os.makedirs(clust_dir)
        idx=list(range(0,len(files)))
        public_photo_indices=[i for i in idx if i not in private_photo_indices]
        for ind in public_photo_indices:
            full_photo = cv2.imread(os.path.join(album_dir, files[ind]))
            r = 200.0 / full_photo.shape[1]
            dim = (200, int(full_photo.shape[0] * r))
            full_photo = cv2.resize(full_photo, dim)
            cv2.imwrite(os.path.join(clust_dir, '%s.jpg' % (files[ind].split("\\")[1])), full_photo)
     




model_desc=[[None,''],['vgg16','_vgg16'],['resnet50','_resnet50'],[None,'_insightface'],[None,'_facenet']]
model_ind=3
if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.txt')
    default_config = config['DEFAULT']
    #print(default_config['InputDirectory'])


    imgProcessing = FacialImageProcessing(print_stat=False, minsize=112)
    process_album(imgProcessing, default_config['InputDirectory'])
    #process_album(imgProcessing, 'Users/lyudakopeikina/Documents/faces')

    imgProcessing.close()

