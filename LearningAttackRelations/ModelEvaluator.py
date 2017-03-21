import ProbabilisticModelsGenerator, GroundedExtensionGenerator, itertools
import numpy as np
from sklearn.metrics import confusion_matrix


def testAnnotatedData(trainingPosts, trainingRatings, testPosts, testRatings):

    highestRating = 10
    positiveArgTypes = [0,3,5] # Consider moving these into a config file
    negativeArgTypes = [1,2,4]
    noArgsTypes = len(positiveArgTypes) + len(negativeArgTypes)

    globalGraph, globalGraphBiDir = ProbabilisticModelsGenerator.getBiDirectionalGlobalGraphs(trainingPosts, trainingRatings)
    predictedRating = []

    for idx, review in enumerate(testPosts):
        posArgs = list(set(positiveArgTypes).intersection(review))
        negArgs = list(set(negativeArgTypes).intersection(review))

        allArgs = sorted(posArgs) + sorted(negArgs)

        bidirIndicesMtxForm = list(itertools.product(posArgs, negArgs))

        argumentGraph = np.zeros(len(allArgs)*len(allArgs))

        # Populate the argument graph using the weights.
        if len(bidirIndicesMtxForm) > 0:

            for bidirAttack in bidirIndicesMtxForm:

                attackPos = (bidirAttack[0]* noArgsTypes) + bidirAttack[1]
                counterPos = (bidirAttack[1]* noArgsTypes) + bidirAttack[0]

                attackProb = globalGraph[attackPos]
                counterProb = globalGraph[counterPos]
                biAttackProb = globalGraphBiDir[attackPos]
                biAttackProb2 = globalGraphBiDir[counterPos]
                allWeights = [attackProb, counterProb, biAttackProb]
                choice = allWeights.index(max(allWeights)) # We take the largest weight

                equivalentNodeAttker = allArgs.index(bidirAttack[0])
                equivalentNodeDefender = allArgs.index(bidirAttack[1])

                if choice == 0:
                    argumentGraph[equivalentNodeAttker * len(allArgs) + equivalentNodeDefender] = 1
                if choice == 1:
                    argumentGraph[equivalentNodeDefender * len(allArgs) + equivalentNodeAttker] = 1
                if choice == 2:
                    argumentGraph[equivalentNodeAttker * len(allArgs) + equivalentNodeDefender] = 1
                    argumentGraph[equivalentNodeDefender * len(allArgs) + equivalentNodeAttker] = 1


        stackedArgGraph = np.vstack( np.array_split(np.array(argumentGraph), len(allArgs)))

        extension = GroundedExtensionGenerator.calculateGroundedExtension(stackedArgGraph)

        allArgsArr = np.array(allArgs)
        winningArguments = list(allArgsArr[extension])

        if len(winningArguments) == 0:
            predictedRating.append('Ntrl')
        if set(winningArguments) <= set(positiveArgTypes):
            predictedRating.append('Pos')
        if set(winningArguments) <= set(negativeArgTypes):
            predictedRating.append('Neg')

    testRatingPolarities = []
    for testRating in testRatings:
        if testRating < 5:
            testRatingPolarities.append('Neg')
        elif testRating > 6:
            testRatingPolarities.append('Pos')
        else:
            testRatingPolarities.append('Ntrl')


    confusionMatrix = confusion_matrix(testRatingPolarities, predictedRating, labels=["Pos", "Ntrl", "Neg"])

    return confusionMatrix



annotatedPosts, numericalRatings = ProbabilisticModelsGenerator.getAnnotations()
noOfFolds = 5 # No of Data folds we will use

annotatedPostsSplit = np.array_split(np.array(annotatedPosts), noOfFolds)
numericalRatingsSplit = np.array_split(np.array(numericalRatings), noOfFolds)

totalRecalls = []
totalPrecisions = []

for fold in range(noOfFolds):
    print('--------Fold ', fold, '---------')
    listOfSplits = list(range(0, noOfFolds))

    testPosts = list(annotatedPostsSplit[fold])
    testRatings = list(numericalRatingsSplit[fold])

    listOfSplits.remove(fold)
    trainingPosts = np.concatenate(np.array(annotatedPostsSplit)[listOfSplits], axis=0)
    trainingRatings = np.concatenate(np.array(numericalRatingsSplit)[listOfSplits], axis=0)


    confusionMatrix = testAnnotatedData(trainingPosts, trainingRatings, testPosts, testRatings)

    print(confusionMatrix)

    sumPredicted = confusionMatrix.sum(axis=0)
    sumActual = confusionMatrix.sum(axis=1)

    recalls = []
    precisions = []

    for idx in range(confusionMatrix.shape[0]):

        recalls.append(confusionMatrix[idx,idx] / sumPredicted[idx])
        precisions.append(confusionMatrix[idx,idx] / sumActual[idx])

    totalRecalls.append(recalls)
    totalPrecisions.append(precisions)

averageRecall = np.mean(np.array(totalRecalls), axis=0)
averagePrecision = np.mean(np.array(totalPrecisions), axis=0)

print('avgRecall: ', averageRecall)
print('avgPrecision: ', averagePrecision)