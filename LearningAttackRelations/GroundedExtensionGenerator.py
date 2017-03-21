import numpy as np
import itertools


def getInOutArgs(argMtx):
    sumArgs = argMtx.sum(axis=0)

    inArgs = np.argwhere(sumArgs == 0)
    inArgs = (inArgs.tolist())
    inArgs = [i[0] for i in inArgs]

    attacked = argMtx[inArgs, :]

    outArgs = (np.unique(np.where(attacked>0)[1])).tolist()

    return inArgs, outArgs

def calculateGroundedExtension(argMtx):
    argTypes = np.array(range(0, argMtx.shape[0]))
    ext = []
    terminate = False

    while not terminate:
        inArgs, outArgs = getInOutArgs(argMtx)

        if len(inArgs) > 0:
            ext.extend(list(argTypes[inArgs]))
            argsDelete = inArgs + outArgs
            argMtx = np.delete(argMtx, argsDelete, axis = 0)
            argMtx = np.delete(argMtx, argsDelete, axis = 1)
            argTypes = np.delete(argTypes, argsDelete)

        else:
            break

        sums = np.sum(argMtx.sum(axis=0))

        # If we find that the resulting graph (having deleted current in and out args) is got no more attacks in it then add
        # whatever is leftover to the extension
        if sums == 0:
            ext.extend(list(argTypes))
            terminate = True

    return ext

def generatePermutations(posArgs, negArgs):

    from itertools import product, chain

    posPerms = np.array(list(itertools.product([0,1], repeat=len(posArgs)*len(negArgs))))
    negPerms = np.array(list(itertools.product([0,1], repeat=len(negArgs)*len(posArgs))))

    allPermsList = [list(chain(*i)) for i in product(posPerms, negPerms)]

    posIdxs = []
    negIdxs = []

    currPos = 0
    for x in range(len(posArgs)):
        start = currPos + len(posArgs)
        posIdxs.extend(range(start, start + len(negArgs)))
        currPos += (len(posArgs) + len(negArgs))


    currNeg = len(posArgs)*(len(posArgs) + len(negArgs))
    for x in range(len(negArgs)):
        start = currNeg
        negIdxs.extend(range(start, start + len(posArgs)))
        currNeg += (len(posArgs) + len(negArgs))

    allPerms = np.zeros(shape=(len(allPermsList), (len(posArgs)+len(negArgs))**2))
    allPermsList = np.array(allPermsList)

    allPerms[:,posIdxs] = allPermsList[:,0:(len(posArgs)*len(negArgs))]
    allPerms[:,negIdxs] = allPermsList[:,(len(posArgs)*len(negArgs)):]

    return allPerms

def calculateProbabilityDistribution(posArgs, negArgs, rating):

    nargs = len(posArgs + negArgs)
    allPermutations = generatePermutations(posArgs, negArgs)

    # Determine the Polarity of the Post
    if rating < 5:
        groundedExtension = negArgs
    elif rating > 6:
        groundedExtension = posArgs
    else:
        groundedExtension = []


    # Iterate over the set of graphs that are possible (excluding circular attacks and same polarity attacks) to see which one's
    # have a grounded extension matching the polarity
    acceptedGraphs = []
    for graph in allPermutations:
        attMtx = np.vstack( np.array_split(np.array(graph), nargs))
        ext = calculateGroundedExtension(attMtx)
        if set(groundedExtension) == set(ext):
            acceptedGraphs.append(graph.tolist())

    # Aggregate then normalise the complete set of Attacks
    acceptedGraphs = np.array(acceptedGraphs)
    sumOfAttacks = (acceptedGraphs).sum(axis=0)
    normalisedAttcks = sumOfAttacks / acceptedGraphs.shape[0]

    return normalisedAttcks


def getAcceptedGraphs(posArgs, negArgs, rating):
    nargs = len(posArgs + negArgs)
    allPermutations = generatePermutations(posArgs, negArgs)

    # Determine the Polarity of the Post
    if rating < 5:
        groundedExtension = negArgs
    elif rating > 6:
        groundedExtension = posArgs
    else:
        groundedExtension = []


    # Iterate over the set of graphs that are possible (excluding circular attacks and same polarity attacks) to see which one's
    # have a grounded extension matching the polarity
    acceptedGraphs = []
    for graph in allPermutations:
        attMtx = np.vstack( np.array_split(np.array(graph), nargs))
        ext = calculateGroundedExtension(attMtx)
        if set(groundedExtension) == set(ext):
            acceptedGraphs.append(graph.tolist())

    # Aggregate then normalise the complete set of Attacks
    acceptedGraphs = np.array(acceptedGraphs)

    return acceptedGraphs