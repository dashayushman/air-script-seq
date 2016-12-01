from tabulate import tabulate
import numpy as np
import os
import markdown2
import csv


# method to pad a vector with and normalize to a fixed length
def padVector(vec, front, back, emg=False):
    if emg:
        return np.lib.pad(vec, (int(front), int(back)), 'constant',
                          constant_values=(0, 0))
    else:
        return np.lib.pad(vec, (int(front), int(back)), 'constant',
                          constant_values=(vec[0], vec[-1]))


def saveMatrixToCsvFile(filepath, mat):
    with open(filepath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(mat)


def createDir(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)


def appendClfReportToListSvm(fileContent, clf_rpt, cm, acc_scr, best_params,
                             itr, n_train, n_test, labels):
    fileContent.append('### Best Params ' + str(best_params))
    fileContent.append('\n')
    fileContent.append('### For nth fold where n = ' + str(itr))
    fileContent.append('\n')
    fileContent.append('### Number of training instances = ' + str(n_train))
    fileContent.append('\n')
    fileContent.append('### Number of validation instances = ' + str(n_test))
    fileContent.append('\n')
    fileContent.append('### Accuracy Score   : ' + str(acc_scr))
    fileContent.append('\n\n')
    fileContent.append('## Classification Report')
    fileContent.append('\n\n')
    fileContent.append(clf_rpt)
    fileContent.append('\n')
    fileContent.append('## Confusion Matrix')
    fileContent.append('\n\n')
    # fileContent.append(tabulate(cm))
    # print(joinMatrix(cm, '\n','\t'))
    fileContent.append(
        tabulate(np.insert(cm, 0, labels, axis=1), labels, tablefmt="html"))

    fileContent.append('\n\n')
    fileContent.append(
        '___________________________________________________________________________________')
    fileContent.append('\n\n')
    return fileContent


def appendClfReportToListNB(fileContent, clf_rpt, cm, acc_scr, algo, itr,
                            n_train, n_test, labels):
    fileContent.append('### Type ' + str(algo))
    fileContent.append('\n')
    fileContent.append('### For nth fold where n = ' + str(itr))
    fileContent.append('\n')
    fileContent.append('### Number of training instances = ' + str(n_train))
    fileContent.append('\n')
    fileContent.append('### Number of validation instances = ' + str(n_test))
    fileContent.append('\n')
    fileContent.append('### Accuracy Score   : ' + str(acc_scr))
    fileContent.append('\n\n')
    fileContent.append('## Classification Report')
    fileContent.append('\n\n')
    fileContent.append(clf_rpt)
    fileContent.append('\n')
    fileContent.append('## Confusion Matrix')
    fileContent.append('\n\n')
    # fileContent.append(tabulate(cm))
    # print(joinMatrix(cm, '\n','\t'))
    fileContent.append(
        tabulate(np.insert(cm, 0, labels, axis=1), labels, tablefmt="html"))

    fileContent.append('\n\n')
    fileContent.append(
        '___________________________________________________________________________________')
    fileContent.append('\n\n')
    return fileContent


def appendClfReportToListKnn(fileContent, clf_rpt, cm, acc_scr, n_neigh, itr,
                             n_train, n_test, labels):
    fileContent.append('### Number of Neighbours ' + str(n_neigh))
    fileContent.append('\n')
    fileContent.append('### For nth fold where n = ' + str(itr))
    fileContent.append('\n')
    fileContent.append('### Number of training instances = ' + str(n_train))
    fileContent.append('\n')
    fileContent.append('### Number of validation instances = ' + str(n_test))
    fileContent.append('\n')
    fileContent.append('### Accuracy Score   : ' + str(acc_scr))
    fileContent.append('\n\n')
    fileContent.append('## Classification Report')
    fileContent.append('\n\n')
    fileContent.append(clf_rpt)
    fileContent.append('\n')
    fileContent.append('## Confusion Matrix')
    fileContent.append('\n\n')
    # fileContent.append(tabulate(cm))
    # print(joinMatrix(cm, '\n','\t'))
    fileContent.append(
        tabulate(np.insert(cm, 0, labels, axis=1), labels, tablefmt="html"))

    fileContent.append('\n\n')
    fileContent.append(
        '___________________________________________________________________________________')
    fileContent.append('\n\n')
    return fileContent


def appendClfReportToListHMM(fileContent, clf_rpt, cm, acc_scr, n_states, itr,
                             n_train, n_test, labels):
    fileContent.append('### Number of States ' + str(n_states))
    fileContent.append('\n')
    fileContent.append('### For nth fold where n = ' + str(itr))
    fileContent.append('\n')
    fileContent.append('### Number of training instances = ' + str(n_train))
    fileContent.append('\n')
    fileContent.append('### Number of validation instances = ' + str(n_test))
    fileContent.append('\n')
    fileContent.append('### Accuracy Score   : ' + str(acc_scr))
    fileContent.append('\n\n')
    fileContent.append('## Classification Report')
    fileContent.append('\n\n')
    fileContent.append(clf_rpt)
    fileContent.append('\n')
    fileContent.append('## Confusion Matrix')
    fileContent.append('\n\n')
    # fileContent.append(tabulate(cm))
    # print(joinMatrix(cm, '\n','\t'))
    fileContent.append(
        tabulate(np.insert(cm, 0, labels, axis=1), labels, tablefmt="html"))

    fileContent.append('\n\n')
    fileContent.append(
        '___________________________________________________________________________________')
    fileContent.append('\n\n')
    return fileContent


def mrkdwn2html(mrkStr):
    return markdown2.markdown(mrkStr)


def getStrFrmList(lst, separator):
    return separator.join(lst)


def joinMatrix(matrix, r_separator, c_separator):
    str_rows = []
    for row in matrix:
        str_rows.append(c_separator.join([str(v) for v in row]))
    return r_separator.join(str_rows)


def appendHeaderToFcListHMM(fileContent, accuracies, header):
    fileContent_h = []
    fileContent_h.append('# For ' + header)
    fileContent_h.append('\n\n')
    fileContent_h.append('## Accuracies for all the folds are as follows\n')
    fileContent_h.append(str(accuracies) + '\n\n')
    fileContent_h.append('## Average Accuracy\n')
    fileContent_h.append(str(np.mean(accuracies)) + '\n\n')
    fileContent_h.append(
        '_______________________________________________________________________________________________________')
    fileContent_h.append('\n\n')
    return fileContent_h + fileContent


def writeToFile(filepath, filecontent):
    with open(filepath, "wt") as out_file:
        out_file.write(filecontent)
