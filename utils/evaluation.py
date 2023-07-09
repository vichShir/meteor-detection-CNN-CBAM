from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    

def accuracy(target, prediction):
    return (target == prediction).sum()/len(target)
    
    
def precision(target, prediction):
    matrix = confusion_matrix(target, prediction)
    tn, fp, fn, tp = matrix.ravel()
    return tp/(tp + fp)


def recall(target, prediction):
    matrix = confusion_matrix(target, prediction)
    tn, fp, fn, tp = matrix.ravel()
    return tp/(tp + fn)


def f1_score(target, prediction):
    rec = recall(target, prediction)
    prec = precision(target, prediction)
    return 2*rec*prec/(rec + prec)


def specificity(target, prediction):
    matrix = confusion_matrix(target, prediction)
    tn, fp, fn, tp = matrix.ravel()
    return tn/(tn + fp)


def sensitivity(target, prediction):
    matrix = confusion_matrix(target, prediction)
    tn, fp, fn, tp = matrix.ravel()
    return tp/(tp + fn) # same as recall


def true_skill_statistic(target, prediction):
    matrix = confusion_matrix(target, prediction)
    tn, fp, fn, tp = matrix.ravel()
    return (tp/(tp+fn))-(fp/(fp+tn))


def false_alarm_ratio(target, prediction):
    matrix = confusion_matrix(target, prediction)
    tn, fp, fn, tp = matrix.ravel()
    return fp/(tp + fp)


def error_rate(target, prediction):
    matrix = confusion_matrix(target, prediction)
    tn, fp, fn, tp = matrix.ravel()
    return (fn + fp) / (tp + fn + fp + tn)