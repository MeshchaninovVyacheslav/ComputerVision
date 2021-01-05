# ============================== 1 Classifier model ============================
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Input, Activation, Dense, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np

def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channgels)
            input shape of image for classification
    :return: nn model for classification
    """
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    return model
    # your code here /\

def fit_cls_model(X, y):
    """
    :param X: 4-dim ndarray with training images
    :param y: 2-dim ndarray with one-hot labels for training
    :return: trained nn model
    """
    # your code here \/
    model = get_cls_model((40, 100, 1))
    model.compile(loss='mse', optimizer=Adam(), metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=100)
    model.save_weights('classifier_model.h5')
    return model
    # your code here /\


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    detection_model = Sequential()
    ind_layer = 0

    kernel = (1, 1)
    while ind_layer < len(cls_model.layers):
        layer = cls_model.layers[ind_layer]
        ind_layer += 1
        print(layer.input_shape, layer.output_shape)
        
        if layer.name.startswith('flatten'):
            Cin = layer.input_shape[3]
            kernel = (layer.input_shape[1], layer.input_shape[2])
            continue
        elif layer.name.startswith('dense'):
            if kernel == (1, 1):
                Cin = layer.input_shape[-1]
            
            new_layer = Conv2D(layer.output_shape[1], kernel, \
                               activation='relu')
            if ind_layer + 1 == len(cls_model.layers):
                new_layer = Conv2D(layer.output_shape[1], kernel, \
                               activation='linear', name='conv2d-'+str(ind_layer))
                               
            new_weights, bias = layer.get_weights()
            new_weights = new_weights.reshape(kernel[0], kernel[1], \
                                                      Cin, layer.output_shape[1]
                                                      )
            new_layer(tf.convert_to_tensor([np.random.rand(kernel[0], kernel[1], Cin)]))
            new_layer.set_weights([new_weights, bias])
            
            kernel = (1, 1)
        else:
            new_layer = layer
            #new_weights = layer.get_weights()
            #new_layer.set_weights(new_weights)
        detection_model.add(new_layer)
    return detection_model
    # your code here /\


def normalize(img):
    return (img - np.min(img)) / (np.max(img) + 1e-10)


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/
    data = np.zeros((len(dictionary_of_images.keys()), 220, 370, 1))
    shapes = []
    for ind, key in enumerate(dictionary_of_images):
        img = dictionary_of_images[key]
        data[ind, 0:img.shape[0], 0:img.shape[1]] = img.reshape((img.shape[0], img.shape[1], 1))
        shapes.append(img.shape)
    y_pred = detection_model.predict(data)[..., 0]
    nums_mp = 2
    dict_ = dict()
    threshold = 0.5
    for ind, key in enumerate(dictionary_of_images):
        detections = []
        proba = normalize(y_pred[ind, 0:shapes[ind][0], 0:shapes[ind][1]])
        for row in range(proba.shape[0]):
            for col in range(proba.shape[1]):
                if proba[row, col] > threshold:
                    detections.append([row*nums_mp, col*nums_mp, 40, 100, proba[row, col]])
        dict_[key] = detections
    return dict_
    # your code here /\

# =============================== 4 Viz ========================================
import matplotlib.patches as patches
def visualize(im, points):

    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im, 'gray')
    for point in points:
        # Create a Rectangle patch
        rect = patches.Rectangle((point[1],point[0]),point[3],point[2],linewidth=1,edgecolor='r',facecolor='none', gid='lol')

        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()

# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    row1, col1, n_rows1, n_cols1 = first_bbox
    row2, col2, n_rows2, n_cols2 = second_bbox
    hieght = max(n_rows1 + n_rows2 - (max(row1 + n_rows1, row2 + n_rows2) - min(row1, row2)), 0)
    width = max(n_cols1 + n_cols2 - (max(col1 + n_cols1, col2 + n_cols2) - min(col1, col2)), 0)
    return (hieght * width) / (n_cols1 * n_rows1 + n_cols2 * n_rows2 - hieght * width + 1e-10)
    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/
    TP = []
    FP = []

    gt_len = 0
    
    for filename in pred_bboxes:
        pred = list(pred_bboxes[filename])
        pred.sort(key=lambda x: -x[-1])
        gt = gt_bboxes[filename]
        gt_len += len(gt)

        for pred_bbox in pred:
            best_bbox = None
            best_iou = 0.5
            for gt_bbox in gt:
                cur_iou = calc_iou(pred_bbox[:-1], gt_bbox)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_bbox = gt_bbox
            if best_bbox is None:
                FP.append(pred_bbox)
            else:
                TP.append(pred_bbox)
                gt.remove(best_bbox)
        
    TPFP = TP + FP
    TPFP.sort(key=lambda x: x[-1])
    TP.sort(key=lambda x: x[-1])
    ind1, ind2 = 0, 0
    res1, res2 = [], []
    for ind1 in range(len(TPFP)):
        while ind2 < len(TP) and TP[ind2][-1] < TPFP[ind1][-1]:
            ind2 += 1
        res1.append(len(TPFP) - ind1)
        res2.append(len(TP) - ind2)
    result = []
    recall, precision = [], []
    for ind1 in range(len(TPFP)):
        pr = res2[ind1] / res1[ind1]
        rec = res2[ind1] / gt_len
        result.append((pr, rec, TPFP[ind1][-1]))
    
    result.sort(key=lambda x: (x[1], -x[0]))
    result = [(1, 0, 0)] + result
    AUC = 0
    for ind in range(len(result) - 1):
        AUC += (result[ind][0] + result[ind + 1][0]) / 2 * np.abs(result[ind + 1][1] - result[ind][1])

    return AUC
    # your code here /\

from copy import deepcopy

# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.5):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    pred_bboxes = deepcopy(detections_dictionary)
    for filename in detections_dictionary:
        pred = list(pred_bboxes[filename])
        pred.sort(key=lambda x: -x[-1])
        ind = 0
        while ind < len(pred) - 1:
            if calc_iou(pred[ind][:-1], pred[ind + 1][:-1]) > iou_thr:
                pred.pop(ind + 1)
            else:
                ind += 1
        pred_bboxes[filename] = pred
    return pred_bboxes
    # your code here /\
