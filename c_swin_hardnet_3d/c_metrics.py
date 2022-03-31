import numpy as np


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def get_score_for_one_patient(labels, predicts, threshold=0.5):
    '''
    计算一个病人的dice、IOU分数
    :param truths: numpy.array, [189, 224, 176, 1]
    :param predicts: numpy.array, [189, 224, 176, 1]
    :param threshold: threshold for computing dice score
    :return: dice score of this patient
    '''
    if labels.shape[0] != 189 or predicts.shape[0] != 189:
        print('ERROR: 每个病人的切片数量应当是189！')
        return 0
    label_positive = labels > threshold
    lp_count = np.count_nonzero(label_positive)
    predict_positive = predicts > threshold
    pp_count = np.count_nonzero(predict_positive)

    TP_count = np.count_nonzero(np.logical_and(label_positive, predict_positive))
    FN_count = lp_count - TP_count
    FP_count = pp_count - TP_count

    dice_score = 2 * TP_count / (lp_count + pp_count) if lp_count + pp_count != 0 else 0
    iou_score = TP_count / (lp_count + pp_count - TP_count) if lp_count + pp_count - TP_count != 0 else 0
    precision = TP_count / (TP_count + FP_count) if FP_count + TP_count != 0 else 0
    recall = TP_count / (TP_count + FN_count) if TP_count + FN_count != 0 else 0
    f1_score = 2 * TP_count / (2 * TP_count + FN_count + FP_count) if 2 * TP_count + FN_count + FP_count != 0 else 0
    voe = 2 * (pp_count - lp_count) / (pp_count + lp_count) if pp_count + lp_count != 0 else 0
    rvd = pp_count / lp_count - 1 if lp_count != 0 else -1

    # print('Label positive:', lp_count,
    #       '\t Predict positive:', pp_count,
    #       '\t TP:', TP_count,
    #       '\t FN:', FN_count,
    #       '\t FP:', FP_count,
    #       '\t Dice score:', dice_score,
    #       '\t IOU score:', iou_score,
    #       '\t Precision:', precision,
    #       '\t Recall:', recall,
    #       '\t F1 score:', f1_score,
    #       '\t VOE score:', voe,
    #       '\t RVD score:', rvd)
    return dice_score, iou_score, precision, recall, f1_score, voe, rvd


def get_score_from_all_slices(labels, predicts, threshold=0.5):
    '''
    输入2维切片，计算每一个病人的3维的分数，返回按照病人计算的平均评价指标。n为切片数量，且须有n%189==0
    :param truths: np.array, [n, 224, 176, 1]
    :param predicts: np.array, [n, 224, 176, 1]
    :param threshold: threshold for computing dice
    :return: a dice scores
    '''
    if labels.shape[0] % 189 != 0 or predicts.shape[0] % 189 != 0:
        print('ERROR: 输入切片数量应当是189的整数倍！')
        return np.array([])
    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    voe_scores = []
    rvd_scores = []
    for i in range(labels.shape[0] // 189):
        tmp_labels = labels[i*189 : (i+1)*189]
        tmp_pred = predicts[i*189 : (i+1)*189]
        tmp_dice, tmp_iou, tmp_precision, tmp_recall, tmp_f1, tmp_voe, tmp_rvd = get_score_for_one_patient(labels=tmp_labels, predicts=tmp_pred, threshold=threshold)
        dice_scores.append(tmp_dice)
        iou_scores.append(tmp_iou)
        precision_scores.append(tmp_precision)
        recall_scores.append(tmp_recall)
        f1_scores.append(tmp_f1)
        voe_scores.append(tmp_voe)
        rvd_scores.append(tmp_rvd)

    scores = {}
    scores['dice'] = dice_scores
    scores['iou'] = iou_scores
    scores['precision'] = precision_scores
    scores['recall'] = recall_scores

    return scores



def get_score_for_one_patient_cherhoo(labels, predicts, threshold=0.5):
    '''
    计算一个病人的dice、IOU分数
    :param truths: numpy.array, [189, 224, 176, 1]
    :param predicts: numpy.array, [189, 224, 176, 1]
    :param threshold: threshold for computing dice score
    :return: dice score of this patient
    '''
    if labels.shape[0] != 189 or predicts.shape[0] != 189:
        print('ERROR: 每个病人的切片数量应当是189！')
        return 0
    label_positive = labels > threshold
    label_negative = labels < threshold
    lp_count = np.count_nonzero(label_positive)   ## 실제 1

    predict_positive = predicts > threshold
    predict_negative = predicts < threshold
    pp_count = np.count_nonzero(predict_positive) ## 예측 1

    TP_count = np.count_nonzero(np.logical_and(label_positive, predict_positive)) ## True Positive(TP) : 실제 True인 정답을 True라고 예측 (정답)
    TN_count = np.count_nonzero(np.logical_and(label_negative, predict_negative)) ## True Negative(TN) : 실제 False인 정답을 False라고 예측 (정답)
    FN_count = lp_count - TP_count  ## False Negative(FN) : 실제 True인 정답을 False라고 예측 (오답)
    FP_count = pp_count - TP_count  ## False Positive(FP) : 실제 False인 정답을 True라고 예측 (오답)

    dice_score = 2 * TP_count / (lp_count + pp_count) if lp_count + pp_count != 0 else 0
    iou_score = TP_count / (lp_count + pp_count - TP_count) if lp_count + pp_count - TP_count != 0 else 0
    precision = TP_count / (TP_count + FP_count) if FP_count + TP_count != 0 else 0
    recall = TP_count / (TP_count + FN_count) if TP_count + FN_count != 0 else 0
    f1_score = 2 * TP_count / (2 * TP_count + FN_count + FP_count) if 2 * TP_count + FN_count + FP_count != 0 else 0
    voe = 2 * (pp_count - lp_count) / (pp_count + lp_count) if pp_count + lp_count != 0 else 0
    rvd = pp_count / lp_count - 1 if lp_count != 0 else -1

    return dice_score, iou_score, precision, recall, f1_score, voe, rvd, [TP_count, TN_count, FN_count, FP_count]


def get_score_from_all_slices_cherhoo(labels, predicts, threshold=0.5):
    '''
    输入2维切片，计算每一个病人的3维的分数，返回按照病人计算的平均评价指标。n为切片数量，且须有n%189==0
    :param truths: np.array, [n, 224, 176, 1]
    :param predicts: np.array, [n, 224, 176, 1]
    :param threshold: threshold for computing dice
    :return: a dice scores
    '''
    if labels.shape[0] % 189 != 0 or predicts.shape[0] % 189 != 0:
        print('ERROR: 输入切片数量应当是189的整数倍！')
        return np.array([])
    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    voe_scores = []
    rvd_scores = []
    metrics_scores = []
    for i in range(labels.shape[0] // 189):
        tmp_labels = labels[i*189 : (i+1)*189]
        tmp_pred = predicts[i*189 : (i+1)*189]
        tmp_dice, tmp_iou, tmp_precision, tmp_recall, tmp_f1, tmp_voe, tmp_rvd, tmp_metrics = get_score_for_one_patient_cherhoo(labels=tmp_labels, predicts=tmp_pred, threshold=threshold)
        dice_scores.append(tmp_dice)
        iou_scores.append(tmp_iou)
        precision_scores.append(tmp_precision)
        recall_scores.append(tmp_recall)
        f1_scores.append(tmp_f1)
        voe_scores.append(tmp_voe)
        rvd_scores.append(tmp_rvd)
        metrics_scores.append(tmp_metrics)

    scores = {}
    scores['dice'] = dice_scores
    scores['iou'] = iou_scores
    scores['precision'] = precision_scores
    scores['recall'] = recall_scores
    scores['f1_scores'] = f1_scores
    scores['metrics'] = metrics_scores

    return scores



if __name__ == '__main__':
    import pandas as pd
    num_fold = 4
    for fold in range(num_fold):
        csv_path = 'fold_' + str(fold) + '/score_record.csv'
        df = pd.read_csv(csv_path)
        print('In fold', fold)
        for key in df.keys():
            print(key, ' = ', np.mean(df[key]))