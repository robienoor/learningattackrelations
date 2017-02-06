import numpy as np
import pandas as pd
import itertools, time

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

def calculateProbabilityDistribution(posArgs, negArgs, rating):
    
    nargs = len(posArgs + negArgs)
    allPermutations = np.array(list(itertools.product([0,1], repeat=nargs*nargs)))
    
    # Determine the Polarity of the Post
    if rating < 5:
        groundedExtension = negArgs
    elif rating > 6:
        groundedExtension = posArgs
    else:
        groundedExtension = []
    
    
    # Create list of Attacks that we will never need. Circular attacks, and attacks between arguments of same polarity
    circularAttacks = (np.arange(0, nargs*nargs, nargs+1)).tolist()
    samePolarityAttacks = []
    posList = list(itertools.permutations(posArgs, 2))
    negList = list(itertools.permutations(negArgs, 2))
    totList = posList + negList

    for l in totList:
        ptn = ((l[0]+1)*nargs)- ((nargs+1) - l[1]) + 1 # All the odd +1 are to account for the shift in 0 index
        samePolarityAttacks.append(ptn)

    
    graphsToDelete = np.unique([circularAttacks + samePolarityAttacks])
    subGraphs = allPermutations[:,graphsToDelete]
    cutDownGraphs = np.delete(allPermutations, (np.where(subGraphs>0)[0]).tolist(), axis = 0)

    # Iterate over the set of graphs that are possible (excluding circular attacks and same polarity attacks) to see which one's 
    # have a grounded extension matching the polarti
    acceptedGraphs = []
    for graph in cutDownGraphs:
        attMtx = np.vstack( np.array_split(np.array(graph), nargs))
        ext = calculateGroundedExtension(attMtx)
        if set(groundedExtension) == set(ext): 
            acceptedGraphs.append(graph.tolist()) 
        
    # Aggregate then normalise the complete set of Attacks
    acceptedGraphs = np.array(acceptedGraphs)
    sumOfAttacks = (acceptedGraphs).sum(axis=0)
    normalisedAttcks = sumOfAttacks / acceptedGraphs.shape[0]
    
    return normalisedAttcks

t0 = time.time()

highestRating = 10
noArgsTypes = 10
positiveArgTypes = [0,1,2]
negativeArgTypes = [3,4,5]
noPosts = 100

randomReviews = np.random.randint(noArgsTypes, size=(noPosts,noArgsTypes))
randomRatings = np.random.randint(highestRating, size=(noPosts,1))

globalGraph = np.zeros(noArgsTypes*noArgsTypes)

for idx, review in enumerate(randomReviews):
    print('-----')
    print('review: ', review)
    print('rating: ', randomRatings[idx])
    
    posArgs = list(set(positiveArgTypes).intersection(review.tolist()))
    print('posArgs: ', posArgs)
    negArgs = list(set(negativeArgTypes).intersection(review.tolist()))
    print('negArgs: ', negArgs)
    posArgsIdx = list(range(0,len(posArgs)))
    negArgsIdx = list(range(len(posArgs), len(posArgs)+len(negArgs)))
    
    nmlDstb = calculateProbabilityDistribution(posArgsIdx, negArgsIdx, randomRatings[idx])
    print(nmlDstb)
    globalContribution = np.zeros(noArgsTypes*noArgsTypes)
    
    noOfArgTypesFound = len(list(posArgs+negArgs))
    
    allAttacks = list(itertools.product(list(posArgs+negArgs), repeat=2))
    allAttacksIdx = []
    for attack in allAttacks:
        ptn = ((attack[0]+1)*noArgsTypes)- ((noArgsTypes+1) - attack[1])
        allAttacksIdx.append(ptn)
    
    globalContribution[allAttacksIdx] = nmlDstb
    
    globalGraph = np.add(globalGraph, globalContribution)
    
 
print('-----------------------------------------------------------------')
print(globalGraph)


x = calculateProbabilityDistribution([0,1], [], 4)



t1 = time.time()
print('time: ', t1-t0)