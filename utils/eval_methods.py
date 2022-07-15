import numpy as np
import sklearn.metrics as skm
from scipy.spatial.distance import directed_hausdorff
import medpy.metric.binary as mmb
import tensorflow as tf
from scipy import ndimage

import SimpleITK as sitk
# evaluation -----------------------------------------------
def softmax(prob_map, axis=-1):
    e = np.exp(prob_map - np.max(prob_map))
    return e / np.sum(e, axis, keepdims=True)
    
def cross_entropy(pred, gt, epsilon=1e-9):
    axis = tuple(range(np.ndim(pred) - 1))# if np.ndim(pred) > 1 else -1
    ce = -np.sum(gt * np.log(pred + epsilon), axis) / pred.shape[0]
    return ce

def mean_squared_error(pred, gt):
    if np.ndim(gt) < 2:
        gt = np.expand_dims(gt, -1) 
    mse = np.mean(np.square(pred - gt), -1)
    return mse

def true_positive(pred, gt):
    axis = tuple(range(1, np.ndim(pred) - 1))# if np.ndim(pred) > 1 else -1
    return np.sum(np.logical_and(pred == gt, gt == 1), axis)

def true_negative(pred, gt):
    axis = tuple(range(1, np.ndim(pred) - 1))# if np.ndim(pred) > 1 else -1
    return np.sum(np.logical_and(pred == gt, gt == 0), axis)

def false_positive(pred, gt):
    axis = tuple(range(1, np.ndim(pred) - 1))# if np.ndim(pred) > 1 else -1
    return np.sum(np.logical_and(pred != gt, pred == 1), axis)

def false_negative(pred, gt):
    axis = tuple(range(1, np.ndim(pred) - 1))# if np.ndim(pred) > 1 else -1
    return np.sum(np.logical_and(pred != gt, pred == 0), axis)

def precision(pred, gt, epsilon=1e-9):
    tp = true_positive(pred, gt)
    fp = false_positive(pred, gt)
    return tp / (tp + fp + epsilon)

def recall(pred, gt, epsilon=1e-9):
    tp = true_positive(pred, gt)
    fn = false_negative(pred, gt)
    return tp / (tp + fn + epsilon)

def sensitivity(pred, gt, epsilon=1e-9):
    return recall(pred, gt, epsilon)

def specificity(pred, gt, epsilon=1e-9):
    tn = true_negative(pred, gt)
    fp = false_positive(pred, gt)
    return tn / (tn + fp + epsilon)

def accuracy(pred, gt):
    """ equal(pred, gt) / all(pred, gt)
        (tp + tn) / (tp + tn + fp + fn)
    """
    axis = tuple(range(1, np.ndim(pred)))# if np.ndim(pred) > 1 else -1
    return np.mean(np.equal(pred, gt), axis)

def dice_coefficient(pred, gt, epsilon=1e-9, ignore_nan=False):
    """ 2 * intersection(pred, gt) / (pred + gt) 
        2 * tp / (2*tp + fp + fn)
    """
    axis = tuple(range(1, np.ndim(pred) - 1))# if np.ndim(pred) > 1 else -1
    intersection = np.sum(pred * gt, axis)
    sum_ = np.sum(pred + gt, axis)
    dice = 2 * intersection / (sum_ + epsilon)
    if ignore_nan:
        no_lab = np.sum(gt, axis) == 0
        dice[no_lab] = np.nan
    return dice

def iou(pred, gt, epsilon=1e-9):
    """ intersection(pred, gt) / union(pred, gt)
        tp / (tp + fp + fn)
    """
    axis = tuple(range(1, np.ndim(pred) - 1))# if np.ndim(pred) > 1 else -1
    intersection = np.sum(pred * gt, axis)
    union = np.sum(pred + gt, axis) - intersection
    return intersection / (union + epsilon)

def hda(pred, gt, epsilon=1e-9):
    lss2 = []
    HDis = []
    HDis1 = []
    lss3 = []

    # gt_ = gt[0, :, :, :,  1]
    # pred_ = pred[0, :, :, :,  1]
    # ans_lab = []
    # for z in range(0, gt_.shape[2]):
    #     for y in range(0, gt_.shape[0]):  # looping through each rows
    #         for x in range(0, gt_.shape[1]):  # looping through each column
    #             if gt_[y, x, z] > 0.5:
    #                 ans_lab.append([y, x, z])
    #
    # ans_pred = []
    # for z in range(0, gt_.shape[2]):
    #     for y in range(0, pred_.shape[0]):  # looping through each rows
    #         for x in range(0, pred_.shape[1]):  # looping through each column
    #             if pred_[y, x, z] > 0.5:
    #                 ans_pred.append([y, x, z])
    # if len(ans_pred) != 0 and len(ans_lab) != 0 :
    #     HDis.append(max(directed_hausdorff(ans_pred, ans_lab)[0], directed_hausdorff(ans_lab, ans_pred)[0]))


    HDis = []

    HDis.append( mmb.hd(pred[0, ...,1], gt[0, ...,1]))
    return np.array(HDis)

def assd(pred, gt, epsilon=1e-9):
    HDis = []
    HDis.append( mmb.assd(pred[0, ..., 1], gt[0, ..., 1]))
    return (np.array(HDis))

    # return np.mean(HDis)
def averaged_hausdorff_distance(set1, set2, max_ahd=np.inf):
    """

    Compute the Averaged Hausdorff Distance function

    between two unordered sets of points (the function is symmetric).

    Batches are not supported, so squeeze your inputs first!

    :param set1: Tensor where each row is an N-dimensional point.

    :param set2: Tensor where each row is an N-dimensional point.

    :return: The Averaged Hausdorff Distance between set1 and set2.

    """
    set1 = tf.convert_to_tensor(set1, dtype=tf.float32)
    set2 = tf.convert_to_tensor(set2, dtype=tf.float32)
 #    assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
 #
 #    assert set2.ndimension() == 2, 'got %s' % set2.ndimension()
 #
 #    assert set1.size()[1] == set2.size()[1], \
 # \
 #        'The points in both sets must have the same number of dimensions, got %s and %s.' \
 # \
 #        % (set2.size()[1], set2.size()[1])


    d2_matrix = cdist(set1, set2)

# Modified Chamfer Loss

    term_1 = tf.reduce_mean(tf.reduce_min(d2_matrix, 1)[0])

    term_2 = tf.reduce_mean(tf.reduce_min(d2_matrix, 0)[0])

    res = term_1 + term_2

    return res

def cdist(x, y):
    """
    Compute distance between each pair of the two collections of inputs.
    :param x: Nxd Tensor
    :param y: Mxd Tensor
    :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
          i.e. dist[i,j] = ||x[i,:]-y[j,:]||

    """
    differences = tf.expand_dims(x, axis =1) - tf.expand_dims(y, axis =0)
    distances = np.sqrt(np.sum(differences**2, -1))
    return distances
def assd_(pred, gt, epsilon=1e-9):
    HDis = [0, 0, 0, 0, 0]
    for i in range(0, pred.shape[0]):
        for j in range(0, pred.shape[-1]):

            HDis[j] = mmb.assd(pred[i, ...,j], gt[i, ...,j])
    return np.reshape(np.array(HDis), [1, 5])


def auc(pred, gt):
    pass
# ----------------------------------------------------------


def Hausdorff_compute(pred,groundtruth,spacing):
    groundtruth = np.transpose(groundtruth, (2, 0, 1))
    pred = np.transpose(pred, (2, 0, 1))

    pred = np.squeeze(pred)
    groundtruth = np.squeeze(groundtruth)

    ITKPred = sitk.GetImageFromArray(pred, isVector=False)
    ITKPred.SetSpacing(spacing)
    ITKTrue = sitk.GetImageFromArray(groundtruth, isVector=False)
    ITKTrue.SetSpacing(spacing)

    overlap_results = np.zeros((1,5, 5))
    surface_distance_results = np.zeros((1,5, 5))

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    for i in range(5):
        pred_i = (pred==i).astype(np.float32)
        if np.sum(pred_i)==0:
            overlap_results[0,i,:]=0
            surface_distance_results[0,i,:]=0
        else:
            # Overlap measures
            overlap_measures_filter.Execute(ITKTrue==i, ITKPred==i)
            overlap_results[0,i, 0] = overlap_measures_filter.GetJaccardCoefficient()
            overlap_results[0,i, 1] = overlap_measures_filter.GetDiceCoefficient()
            overlap_results[0,i, 2] = overlap_measures_filter.GetVolumeSimilarity()
            overlap_results[0,i, 3] = overlap_measures_filter.GetFalseNegativeError()
            overlap_results[0,i, 4] = overlap_measures_filter.GetFalsePositiveError()
            # Hausdorff distance
            #hausdorff_distance_filter.Execute(ITKTrue==i, ITKPred==i)

            #surface_distance_results[0,i, 0] = hausdorff_distance_filter.GetHausdorffDistance()
            # Symmetric surface distance measures

            reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKTrue == i, squaredDistance=False, useImageSpacing=True))
            reference_surface = sitk.LabelContour(ITKTrue == i)
            statistics_image_filter = sitk.StatisticsImageFilter()
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(reference_surface)
            num_reference_surface_pixels = int(statistics_image_filter.GetSum())

            segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKPred==i, squaredDistance=False, useImageSpacing=True))
            segmented_surface = sitk.LabelContour(ITKPred==i)
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(segmented_surface)
            num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

            # Multiply the binary surface segmentations with the distance maps. The resulting distance
            # maps contain non-zero values only on the surface (they can also contain zero on the surface)
            seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
            ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

            # Get all non-zero distances and then add zero distances if required.
            seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
            seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
            seg2ref_distances = seg2ref_distances + \
                                list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
            ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
            ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
            ref2seg_distances = ref2seg_distances + \
                                list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

            all_surface_distances = seg2ref_distances + ref2seg_distances

            # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
            # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
            # segmentations, though in our case it is. More on this below.
            surface_distance_results[0,i, 1] = np.mean(all_surface_distances)
            surface_distance_results[0,i, 2] = np.median(all_surface_distances)
            surface_distance_results[0,i, 3] = np.std(all_surface_distances)
            surface_distance_results[0,i, 4] = np.max(all_surface_distances)


    return overlap_results,surface_distance_results