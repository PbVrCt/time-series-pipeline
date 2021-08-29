"""
'Pruning' as shown in Marcos Lopez de Prado "Advances in Financial Machine Learning" Chapter 7

For cases  when the test set is not the most recent one in time, such as cross validation, another technique
shown in that chapter, that the author calls 'Embargo', would also be useful. It would help, at least, to prevent the data leakage
from the test to the training set that would originate from trailing window standarization
"""
# 'Pruning'
def getTrainTimes(t1train, t1test):
    """
    Removes instances of the train set that overlap with the test set
    —t1.index: timestamps of the features
    —t1.value: timestamps of the labels
    """
    trn = t1train.copy(deep=True)
    for i, j in t1test.iteritems():
        # test features precede train features and train features precede test labels: train starts within test
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index  #
        # test features precede train labels and train labels precede test labels: train ends within test
        df1 = trn[(i <= trn) & (trn <= j)].index  #
        # train features precede test features and test labels precede train labels: train envelops test
        df2 = trn[(trn.index <= i) & (j <= trn)].index
        trn = trn.drop(df0.union(df1).union(df2))
    return trn
