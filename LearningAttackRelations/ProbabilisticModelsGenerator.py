import json, itertools, math
import GroundedExtensionGenerator
import numpy as np

highestRating = 10
positiveArgTypes = [0,3,5] # Consider moving these into a config file
negativeArgTypes = [1,2,4]
noArgsTypes = len(positiveArgTypes) + len(negativeArgTypes)

urlAnnotations = 'allAnnotationsChutesRun1.json'

def getAnnotations():
    # Collect the annotation data into a numpy array
    annotatedPosts = []
    numericalsRatings = []

    with open(urlAnnotations) as data_file:
        with open('ForumPosts.json') as ratings_file:
            data = json.load(data_file)


            ratings = json.load(ratings_file)

            for idx, d in enumerate(data):
                # We get 'Nones' sometimes when things left completeley blank. Replace with 0's so as not to interfere
                # with our sums whilst letting us keep track of noOfSentences per post
                d = [[0, 0, 0, 0, 0, 0, 0] if v is None else v for v in d]
                d = np.array(d)
                sums = d.sum(axis=0)
                sums = np.argwhere(sums > 0)
                sums = sums[sums !=6] # We ignore the annotations for the last category (6th - other) as we do not know how to use it in the argument graph or its polarity properly
                annotatedPosts.append(sums.tolist())
                numericalsRatings.append(ratings[idx]['Rating'])

    return annotatedPosts, numericalsRatings


def getSingularAttackProbModelDev(acceptedGraphs):
    sumOfAttacks = acceptedGraphs.sum(axis=0)
    return sumOfAttacks / acceptedGraphs.shape[0]


def getBiDirectionalAttackModel(acceptedGraphs, posArgsIdx, negArgsIdx):

    # Calculate the indices of those positions that we need to look at to see if there is a bidirectional attack.
    bidirIndicesMtxForm = list(itertools.product(posArgsIdx,negArgsIdx)) # This is a list of positions in which there might be a bidirectional attack for a square matrix
    bidirIndices = []
    noOfArgs = len(posArgsIdx) + len(negArgsIdx)

    nmlDistributionBiDir = np.zeros(noOfArgs * noOfArgs)

    for r, c in bidirIndicesMtxForm:
        vectorPs = r*(len(posArgsIdx+negArgsIdx)) + c # This is us converting the matrix coordinates to a vector coordinate
        vectorPsOpp = c*(len(posArgsIdx+negArgsIdx)) + r # This is the diagnol opposite, meaning the location where the bidirectional attack may be
        bidirIndices.append((vectorPs, vectorPsOpp)) #Here we find the equivalent positions if the square matrix where reshaped into a vector

        biLocations = acceptedGraphs[:,vectorPs] + acceptedGraphs[:, vectorPsOpp]
        biLocations = np.argwhere(biLocations == 2)


        if biLocations.shape[0] > 0:
            acceptedGraphs[biLocations, [vectorPs,vectorPsOpp]] = 0
            nmlDistributionBiDir[[vectorPs,vectorPsOpp]] = biLocations.shape[0]


    sumOfAttacks = acceptedGraphs.sum(axis=0)
    nmlDistribution = sumOfAttacks / acceptedGraphs.shape[0]
    nmlDistributionBiDir = nmlDistributionBiDir / acceptedGraphs.shape[0]
    return nmlDistribution, nmlDistributionBiDir


def getSingleDirectionGlobalGraph(annotatedPosts, numericalsRatings):

    # Begin generating the distributions per post
    globalGraph = np.zeros(noArgsTypes*noArgsTypes)

    for idx, review in enumerate(annotatedPosts):

        posArgs = list(set(positiveArgTypes).intersection(review))
        negArgs = list(set(negativeArgTypes).intersection(review))
        posArgsIdx = list(range(0,len(posArgs)))
        negArgsIdx = list(range(len(posArgs), len(posArgs)+len(negArgs)))

        acceptedGraphs = GroundedExtensionGenerator.getAcceptedGraphs(posArgsIdx, negArgsIdx, numericalsRatings[idx])
        nmlDstb = getSingularAttackProbModelDev(acceptedGraphs)
        globalContribution = np.zeros(noArgsTypes*noArgsTypes)

        allAttacks = list(itertools.product(list(posArgs+negArgs), repeat=2)) # Generate the indexes for the attack relations. We use these to make sure we place our nmldstrbution into the correct position in the global graph
        allAttacksIdx = []
        for attack in allAttacks:
            ptn = ((attack[0]+1)*noArgsTypes)- ((noArgsTypes+1) - attack[1])+1
            allAttacksIdx.append(ptn)


        globalContribution[allAttacksIdx] = nmlDstb

        if np.isnan(globalContribution).any():
            continue

        globalGraph = np.add(globalGraph, globalContribution)
        #print(globalGraph)

    globalGraphMtxForm = np.vstack( np.array_split(np.array(globalGraph), noArgsTypes))

    return globalGraph


def getBiDirectionalGlobalGraphs(annotatedPosts, numericalsRatings): # Developed to test the bidirectional graphs

    # Begin generating the distributions per post
    globalGraph = np.zeros(noArgsTypes*noArgsTypes)
    globalGraphBiDir = np.zeros(noArgsTypes*noArgsTypes)

    for idx, review in enumerate(annotatedPosts):

        posArgs = list(set(positiveArgTypes).intersection(review))
        negArgs = list(set(negativeArgTypes).intersection(review))
        posArgsIdx = list(range(0,len(posArgs)))
        negArgsIdx = list(range(len(posArgs), len(posArgs)+len(negArgs)))

        acceptedGraphs = GroundedExtensionGenerator.getAcceptedGraphs(posArgsIdx, negArgsIdx, numericalsRatings[idx])
        nmlDstb, nmlDstbBiDir = getBiDirectionalAttackModel(acceptedGraphs, posArgsIdx, negArgsIdx)

        globalContribution = np.zeros(noArgsTypes*noArgsTypes)
        globalContributionBiDir = np.zeros(noArgsTypes*noArgsTypes)

        allAttacks = list(itertools.product(list(posArgs+negArgs), repeat=2)) # Generate the indexes for the attack relations. We use these to make sure we place our nmldstrbution into the correct position in the global graph

        allAttacksIdx = []
        for attack in allAttacks:
            ptn = ((attack[0]+1)*noArgsTypes)- ((noArgsTypes+1) - attack[1])+1
            allAttacksIdx.append(ptn)


        globalContribution[allAttacksIdx] = nmlDstb
        globalContributionBiDir[allAttacksIdx] = nmlDstbBiDir

        if np.isnan(globalContribution).any():
            print('nan in single attacks')
            continue

        globalGraph = np.add(globalGraph, globalContribution)
        globalGraphBiDir = np.add(globalGraphBiDir, globalContributionBiDir)


        globalGraphMtxForm = np.vstack( np.array_split(np.array(globalGraph), noArgsTypes))
        globalGraphMtxFormBidir = np.vstack( np.array_split(np.array(globalGraphBiDir), noArgsTypes))


    return globalGraph, globalGraphBiDir

