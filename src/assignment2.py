import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, pairwise
from collections import Counter
import matplotlib.pyplot as plt

# Read in the feature data
def ReadInputData(fileName):
    # Transpose so the rows are the documents and the features are the columns 
    return np.loadtxt(fileName).transpose()

# Read in the class data
def ReadClassData(fileName):
    # Only get the column with the classifications
    return np.loadtxt(fileName)[:, 1] 

# Gets top k Cosine Similarity values (nearest neighbor training document classes) for each test document
def GetNearestNeighborClasses(xTrain, xTest, yTrain, k):
    # Calculate Euclidian distance
    cosineSimilarityMatrix = pairwise.cosine_similarity(X=xTest, Y=xTrain)
    maxValues = []
    # For each test document
    for testDocument in cosineSimilarityMatrix:
        # Get the training documents (indicies) of the k nearest neighbors
        indicies = np.argsort(testDocument)[-k:]
        # Find the corresponding classes
        classes = [yTrain[i] for i in indicies]
        maxValues.append(classes)
    return np.array(maxValues)

# Does all calculations for K Nearest Neighbors classifier
def RunKNearestNeighbors(xTrain, yTrain, xTest, yTest, k):
    yPredicted = []
    # Find the k nearest neighbor training document classes of every test document
    knn = GetNearestNeighborClasses(xTrain, xTest, yTrain, k)
    # For each list of test document nearest neighbors
    for testDocumentNN in knn:
        # Find the most frequent class of the nearest neighbors and make that the prediction
        values, counts = np.unique(testDocumentNN, return_counts=True)
        yPredicted.append(values[counts.argmax()])
    # Calculate confusion matrix and accuracy
    cm = confusion_matrix(yTest, yPredicted)
    accuracy = (cm[0][0] + cm[1][1]) / 200
    p = cm[1][1] / (cm[1][1] + cm[1][0])
    r = cm[1][1] / (cm[1][1] + cm[0][1])
    f1 = 2 * (p * r) / (p + r)
    return accuracy, cm, p, r, f1

# Does all calculations for Multinomial Naive Bayes classifier
def RunSkLearnMultinomialNB(xTrain, yTrain, xTest, yTest):
    # Create classifier
    nb = MultinomialNB()
    # Train classifier
    nb.fit(xTrain,yTrain)
    # Test classifier
    yPredicted = nb.predict(xTest)
    # Calculate confusion matrix and accuracy
    cm = confusion_matrix(yTest, yPredicted)
    accuracy = (cm[0][0] + cm[1][1]) / 200
    p = cm[1][1] / (cm[1][1] + cm[1][0])
    r = cm[1][1] / (cm[1][1] + cm[0][1])
    f1 = 2 * (p * r) / (p + r)
    return accuracy, cm, p, r, f1

# Does all calculations for SVM classifier
def RunSkLearnSVM(xTrain, yTrain, xTest, yTest):
    # Create classifier
    svm = LinearSVC()
    # Train classifier
    svm.fit(xTrain,yTrain)
    # Test classifier
    yPredicted = svm.predict(xTest)
    # Calculate confusion matrix and accuracy
    cm = confusion_matrix(yTest, yPredicted)
    accuracy = (cm[0][0] + cm[1][1]) / 200
    p = cm[1][1] / (cm[1][1] + cm[1][0])
    r = cm[1][1] / (cm[1][1] + cm[0][1])
    f1 = 2 * (p * r) / (p + r)
    return accuracy, cm, p, r, f1

# Write confusion matrix, accuracy, precision, recall, and f1 data to file
def WriteStatsToFile(fileName, accuacy, confusionMatrix, p, r, f1):
    fileObject = open(fileName, "w")
    fileObject.write("Accuracy:" + str(accuacy) + "\n")
    fileObject.write("Precision:" + str(p) + "\n")
    fileObject.write("Recall:" + str(r) + "\n")
    fileObject.write("F1:" + str(f1) + "\n")
    fileObject.write("Confusion Matrix:\n")
    for row in confusionMatrix:
        fileObject.write(" ".join([str(a) for a in row]) + "\n")
    fileObject.close()

# Graph class distribution in the training and test sets
def GraphClassData(yTrain, yTest):
    trainCounter = Counter(yTrain)
    testCounter = Counter(yTest)
    sets = ("Train", "Test")
    counts = {
        'Microsoft Windows': (trainCounter[0.0], testCounter[0.0]),
        'Hockey': (trainCounter[1.0], testCounter[1.0])
    }

    x = np.arange(len(sets))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in counts.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution in the Training and Test Sets')
    ax.set_xticks(x + width, sets)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 500)

    # Uncomment to show graph
    plt.show()

################### MAIN CODE ############################
# Read in files
xTrain = ReadInputData("./2Newsgroups/trainMatrixModified.txt")
yTrain = ReadClassData("./2Newsgroups/trainClasses.txt")
xTest = ReadInputData("./2Newsgroups/testMatrixModified.txt")
yTest = ReadClassData("./2Newsgroups/testClasses.txt")

# Graph class distribution
GraphClassData(yTrain, yTest)

# Run classifiers
nbAc, nbMat, nbP, nbR, nbF1 = RunSkLearnMultinomialNB(xTrain, yTrain, xTest, yTest)
svmAc, svmMat, svmP, svmR, svmF1 = RunSkLearnSVM(xTrain, yTrain, xTest, yTest)
knnAc, knnMat, knnP, knnR, knnF1 = RunKNearestNeighbors(xTrain, yTrain, xTest, yTest, k=5) 

# Write classifier stats to files
WriteStatsToFile("NB.txt", nbAc, nbMat, nbP, nbR, nbF1)
WriteStatsToFile("SVM.txt", svmAc, svmMat, svmP, svmR, svmF1)
WriteStatsToFile("KNN.txt", knnAc, knnMat, knnP, knnR, knnF1)