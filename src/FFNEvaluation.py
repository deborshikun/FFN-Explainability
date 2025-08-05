import numpy as np
import time
import random
import numpy
import onnx
import onnxruntime as rt
from random import randint
import sys
sys.path.append('..')

from src.vnnlib import readVnnlib, getIoNodes
from src.util import predictWithOnnxruntime, removeUnusedInitializers
from src.util import  findObjectiveFuncionType, checkAndSegregateSamplesForMaximization, checkAndSegregateSamplesForMinimization

'Number of times FFN runs'
numRuns=100

'Number of samples in each FFN run'
numSamples=150

supersats = {}
superunsats = {}


def onnxEval(onnxModel,inVals,inpDtype, inpShape):
   flattenOrder='C'
   inputs = np.array(inVals, dtype=inpDtype)
   inputs = inputs.reshape(inpShape, order=flattenOrder) # check if reshape order is correct
   assert inputs.shape == inpShape

   output = predictWithOnnxruntime(onnxModel, inputs)
   flatOut = output.flatten(flattenOrder) # check order, 'C' for row major order
   return flatOut

def propCheck(inputs,specs,outputs):
   res = "unknown"
   i=0
   k=0
   flag_for_advinput_found = 0
   'Check for each property in specs'
   'If any property is violated, returns the inputs and outputs'
   'If no property is violated, returns empty supersats'
   for propMat, propRhs in specs:
        vec = propMat.dot(outputs)
        sat = np.all(vec <= propRhs)
        if sat:
          #   res = 'violated'
          #   print("\nProperty violated for inputs - ", inputs)
           supersats[i] = inputs
           flag_for_advinput_found = 1
           print("Input Space checked: ", len(list(supersats.values())) + len(list(superunsats.values())))
           print("Adversarial inputs found - ", list(supersats.values()))
           print("Number of adversarial inputs found - ", len(supersats))
           i += 1
           #return flag_for_advinput_found, supersats  # Return the adversarial inputs found
        else:
             superunsats[k] = inputs
             k += 1

           #return flag_for_advinput_found, supersats  # Return the adversarial inputs found
           #return 1
   return flag_for_advinput_found , supersats
   #return 0
    

def learning(cpos,cneg,iRange,numInputs):
    for i in range (len(cneg)):
        nodeSelect = randint(0,int(numInputs)-1)
        cp=cpos[0][0]
        cn=cneg[i][0]
        cposInVal=cp[nodeSelect]
        cnegInVal=cn[nodeSelect]
        if( cposInVal > cnegInVal):
            temp = round(random.uniform(cnegInVal, cposInVal), 6)
            if ( temp <= iRange[nodeSelect][1] and temp >= iRange[nodeSelect][0]):
                iRange[nodeSelect][0]=temp
        else:
            if (cposInVal < cnegInVal):
                temp = round(random.uniform(cposInVal, cnegInVal), 6)
                if ( temp <= iRange[nodeSelect][1] and temp >= iRange[nodeSelect][0]):
                   iRange[nodeSelect][1]=temp


def makeSample(onnxModel,numInputs,inRanges,samples,specs,inpDtype, inpShape):
    sampleInputList=[]
    
    'Generates all samples and strored in a list (sampleInputList)'
    for k in range(numSamples):
        
        'Checking for duplicate sample values'
        'If duplicate found, generates another sample value'
        'This checking is done for 5 times with a hope that it will generate a new one within these 5 times'
        j=0
        while (j<5):
            inValues=[]
            for i in range(numInputs):
                inValues.append(round(random.uniform(inRanges[i][0],inRanges[i][1]),6))

            '''Check for "Duplicate" samples
               If new samples are in sampleInputList then continues to get other samples
            '''
            if ( inValues in sampleInputList):
                #print("Duplicate")
                j=j+1 #check needed
            else :
                break
        sampleInputList.append(inValues)

        #onnx model evaluaion with new sampled inputs
        sampleVal=onnxEval(onnxModel,inValues,inpDtype, inpShape)
        #checking property with onnx evaluation outputs
        retVal,advinps=propCheck(inValues,specs,sampleVal)

        if retVal == 1: #Adversarial found
            return 1 , advinps
            #return 1
        
        s = []
        s.append(inValues)
        s.append(sampleVal)
        samples.append(s)
    return 0 , 0 #No adversarial found
    #return 0

def runSample(onnxModel, numInputs, numOutputs, inputRange, tAndOT, spec, inpDtype, inpShape):
    oldPosSamples = []
    target = tAndOT[0]
    objectiveType = tAndOT[1]
    all_advinps = []

    'Run FFN for numRuns'
    for k in range(numRuns):
        samples = []
        posSamples = []
        negSamples = []

        ret, advinps = makeSample(onnxModel, numInputs, inputRange, samples, spec, inpDtype, inpShape)

        # If adversarial inputs found, accumulate them
        if ret == 1 and advinps:
            # advinps is a dict, get all values
            all_advinps.extend(list(advinps.values()))

        'Segregate sample list into psitive and negative samples according to the objective type found'
        if ( objectiveType == 1) :
           checkAndSegregateSamplesForMinimization(posSamples,negSamples,samples,oldPosSamples,target)
        else:
           checkAndSegregateSamplesForMaximization(posSamples,negSamples,samples,oldPosSamples,target)
        oldPosSamples = posSamples

        'Check input ranges for further sampling'
        'Discontinues if all the input ranges are below a theshold value(0.000001)'
        flag = False
        for i in range(numInputs):
            if ( inputRange[i][1] - inputRange[i][0] > 0.000001):
               flag = True
               break

        if( flag == False):
           #print("!!! No further sampling Possible for this iteration!!!")
           #print("Inputs are now :: ", inputRange)
           #print("STATUS :: unknown")
           return "unknown", all_advinps

        learning(posSamples,negSamples,inputRange,numInputs)
    return "timeout", all_advinps


#SampleEval function
def sampleEval(onnxFilename, vnnlibFilename):
    onnxModel = onnx.load(onnxFilename)
    onnx.checker.check_model(onnxModel, full_check=True)
    onnxModel = removeUnusedInitializers(onnxModel)
    inp, out, inpDtype = getIoNodes(onnxModel)
    inpShape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
    outShape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in out.type.tensor_type.shape.dim)

    numInputs = 1
    numOutputs = 1
    for n in inpShape:
        numInputs *= n
    for n in outShape:
        numOutputs *= n

    boxSpecList = readVnnlib(vnnlibFilename, numInputs, numOutputs)
    targetAndType = findObjectiveFuncionType(boxSpecList[0][1], numOutputs)

    all_advinps = []
    returnStatus = "timeout"

    for i in range(len(boxSpecList)):
        boxSpec = boxSpecList[i]
        inRanges = boxSpec[0]
        specList = boxSpec[1]
        random.seed()
        returnStatus, advinps = runSample(onnxModel, numInputs, numOutputs, inRanges, targetAndType, specList, inpDtype, inpShape)
        all_advinps.extend(advinps)
        if returnStatus == "violated":
            # Continue collecting, do not return early
            pass
    return returnStatus, all_advinps




