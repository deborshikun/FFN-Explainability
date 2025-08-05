import sys
import time
import signal
import argparse

from src.FFNEvaluation import sampleEval

# Register a handler for the timeout
def handler(signum, frame):
    raise Exception("")  # kill running :: Timeout occurs
def runSingleInstanceForAllCategory(onnxFile,vnnlibFile,timeout):
   'called from run_all_catergory.py'

   # Register the signal function handler
   signal.signal(signal.SIGALRM, handler)

   # Define a timeout for "runSingleInstance"
   signal.alarm(int(timeout))

   '"runSingleInstance" will continue until any adversarial found or timeout occurs'
   'When timeout occurs codes written within exception will be executed'
   try:
       retStatus = runSingleInstance(onnxFile,vnnlibFile)
       return retStatus
   except Exception as exc:
       #printStr = "timeout," + str(timeout) + "\n" 
       print(exc)


def runSingleInstance(onnxFile, vnnlibFile):
    # Variable Initialization
    startTime = time.time()

    onnxFileName = onnxFile.split('/')[-1]
    vnnFileName = vnnlibFile.split('/')[-1]

    print(f"\nTesting network model {onnxFileName} for property file {vnnFileName}")

    # Calling sampleEval until any adversarial found or timeout occurs
    while True:
        # 🆕 sampleEval now returns both status and adversarial input
        status, adv_input = sampleEval(onnxFile, vnnlibFile)
        endTime = time.time()
        timeElapsed = endTime - startTime

        if status == "violated":
            # 🆕 Format includes time and adversarial input
            resultStr = f"{status}, {round(timeElapsed, 4)}, adv_input: {adv_input}"
            return resultStr

    # If never violated and no timeout occurs, it would run forever (timeout handled outside)


# Main function
if __name__ == '__main__':

    # Commandline arguments processing
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('-m', help='A required onnx model file path')
    parser.add_argument('-p', help='A required vnnlib file path')
    parser.add_argument('-o', help='An optional result file path')
    parser.add_argument('-t', help='An optional timeout')

    args = parser.parse_args()
    onnxFile = args.m
    vnnlibFile = args.p

    if onnxFile is None:
        print("\n!!! Failed to provide onnx file on the command line!")
        sys.exit(1)

    if vnnlibFile is None:
        print("\n!!! Failed to provide vnnlib file path on the command line!")
        sys.exit(1)

    resultFile = args.o
    if resultFile is None:
        resultFile = "out.txt"
        print("\n!!! No result_file path is provided on the command line!")
        print(f"Output will be written in default result file- \"{resultFile}\"")
    else:
        print(f"\nOutput will be written in - \"{resultFile}\"")

    timeout = args.t
    if timeout is None:
        print("\n!!! timeout is not on the command line!")
        print("Default timeout is set as - 60 sec")
        timeout = 60.0
    else:
        timeout = float(timeout)  # 🆕 Ensure it's float
        print(f"\ntimeout is  - {timeout} sec")

    # Register the signal function handler
    signal.signal(signal.SIGALRM, handler)

    # Define a timeout for "runSingleInstance"
    signal.alarm(int(timeout))

    # "runSingleInstance" will continue until any adversarial found or timeout occurs
    outFile = open(resultFile, "w")
    try:
        retStatus = runSingleInstance(onnxFile, vnnlibFile)
        print(f"\nOutput is written in - \"{resultFile}\"")

    except Exception as exc:
        print(f"\nOutput is written in - \"{resultFile}\"")
        retStatus = f"timeout,{timeout}\n"
        print(exc)

    outFile.write(retStatus)
    outFile.close()
