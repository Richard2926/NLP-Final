# DO NOT IMPORT ANYTHING IN THIS FILE. You shouldn't need any external libraries.

# accuracy
#
# What percent of classifications are correct?
# 
# true: ground truth, Python list of booleans.
# pred: model predictions, Python list of booleans.
# return: percent accuracy bounded between [0, 1]
#
def accuracy(true, pred):
    acc = None
    ## YOUR CODE STARTS HERE (~2-5 lines of code) ##
    # print(len(true))
    # print(len(pred))
    # print(true)
    # print(pred)
    corr = 0
    for i in range(len(true)):
      if true[i] == pred[i]:
        corr += 1
    acc = corr / len(true)
    ## YOUR CODE ENDS HERE ##
    return acc

# binary_f1 
#
# A method to calculate F-1 scores for a binary classification task.
# 
# args -
# true: ground truth, Python list of booleans.
# pred: model predictions, Python list of booleans.
# selected_class: Boolean - the selected class the F-1 
#                 is being calculated for.
# 
# return: F-1 score between [0, 1]
#
def binary_f1(true, pred, selected_class=True):
    f1 = None
    ## YOUR CODE STARTS HERE (~10-15 lines of code) ##
    truepos, falsepos, falseneg = 0, 0, 0
    # print(true)
    # print(pred)
    for i in range(len(true)):
      if(true[i] == selected_class and true[i] == pred[i]):
        # print("truepos")
        truepos += 1
      elif (pred[i] != selected_class and true[i] == selected_class):
        falseneg += 1
        # print("falseneg")
      elif (pred[i] == selected_class and true[i] != selected_class):
        falsepos +=1
      # print(true[i])
      # print(pred[i])
    # print(truepos)
    # print(falsepos)
    # print(falseneg)
    if(truepos + falsepos != 0):
      precision = truepos/(truepos + falsepos)
    else:
      precision = 1
    if(truepos + falseneg != 0):
      recall = truepos/(truepos+falseneg)
    else:
      recall = 1

    if(precision + recall != 0):
      f1 = (2*precision*recall)/(precision + recall)
    else:
      f1 = 0

    ## YOUR CODE ENDS HERE ##
    return f1

# binary_macro_f1
# 
# Averaged F-1 for all selected (true/false) clases.
#
# args -
# true: ground truth, Python list of booleans.
# pred: model predictions, Python list of booleans.
#
#
def binary_macro_f1(true, pred):
    averaged_macro_f1 = None
    ## YOUR CODE STARTS HERE (1 line of code) ##
    # print(true)
    # print(pred)
    averaged_macro_f1 = (binary_f1(true, pred, True) + binary_f1(true, pred, False))/2
    ## YOUR CODE ENDS HERE ##
    return averaged_macro_f1
