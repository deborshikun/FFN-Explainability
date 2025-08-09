import sys
import time
import signal_windows as signal #modified for windows
import argparse

from src.FFNEvaluation import sampleEval, supersats, superunsats

# Global variable for timeout
system_timeout = 60.0  # Default value

# Register an handler for the timeout
def handler(signum, frame):
    raise Exception("")#kill running :: Timeout occurs")

def runSingleInstanceForAllCategory(onnxFile, vnnlibFile, timeout_val):
   'called from run_all_catergory.py'
   
   # Use a local timeout value
   local_timeout = float(timeout_val)
   
   try:
       retStatus = runSingleInstance(onnxFile, vnnlibFile, local_timeout)
       return retStatus
   except Exception as exc:
       print(exc)
       return "timeout," + str(local_timeout) + "\n"

def runSingleInstance(onnxFile, vnnlibFile, timeout_duration=None):
   # If no timeout provided, use the system default
   if timeout_duration is None:
       timeout_duration = system_timeout
   
   # Variable Initialization
   startTime = time.time()
   all_adv_inputs = []
   
   onnxFileName = onnxFile.split('/')[-1]
   vnnFileName = vnnlibFile.split('/')[-1]
   
   # Calculate end time based on provided timeout
   end_time = startTime + float(timeout_duration)
   
   # print(f"\nTesting network model {onnxFileName} for property file {vnnFileName}")
   
   # Run until the full timeout period is used
   iteration = 0
   while time.time() < end_time:
       iteration += 1
      #  print(f"Starting iteration {iteration}, remaining time: {end_time - time.time():.2f}s")
       
       # Call sampleEval to find adversarial inputs
       status, adv_inputs = sampleEval(onnxFile, vnnlibFile)
       
       # Add any found adversarial inputs to our collection
       if adv_inputs and len(adv_inputs) > 0:
           for inp in adv_inputs:
               if inp not in all_adv_inputs:
                   all_adv_inputs.append(inp)
                  #  print(f"Found new adversarial input: {inp}")
   
   # Calculate total time used
   timeElapsed = time.time() - startTime
   
   # Prepare result string
   result = "" #f"Testing network model {onnxFileName} for property file {vnnFileName}\n"
   
   if all_adv_inputs:
       result += f"Status: violated\n"
      #  result += f"Time elapsed: {timeElapsed:.4f} seconds\n"
      #  result += f"Total iterations completed: {iteration}\n"
      #  result += f"Input Space checked: {len(supersats) + len(superunsats)}\n"
       result += f"Adversarial inputs found: {all_adv_inputs}\n"
   else:
       result += f"Status: timeout\n"
      #  result += f"Time elapsed: {timeElapsed:.4f} seconds\n"
      #  result += f"Total iterations completed: {iteration}\n"
   
   return result


#Main function
if __name__ == '__main__':
   # Parse arguments
   parser = argparse.ArgumentParser()

   # Required onnx file path 
   parser.add_argument('-m',
                    help='A required onnx model file path')

   # Required vnnlib file path
   parser.add_argument('-p', 
                    help='A required vnnlib file path')

   # Optional resultfile path
   parser.add_argument('-o',
                    help='An optional result file path')

   # optional timeout parameter
   parser.add_argument('-t',
                    help='An optional timeout')

   args = parser.parse_args()
   onnxFile = args.m
   vnnlibFile = args.p
   
   'Check for the onnxfile in the commandline, it is a mandatory parameter'
   if (onnxFile is None):
      print ("\n!!! Failed to provide onnx file on the command line!")
      sys.exit(1)  # Exit from program

   'Check for the vnnlib file in the commandline, it is a mandatory parameter'
   if (vnnlibFile is None):
      print ("\n!!! Failed to provide vnnlib file path on the command line!")
      sys.exit(1)  # Exit from program


   resultFile = args.o 

   'Set default for resultFile if no result file is provided in the commandline'
   'It is an optional parameter'
   if ( resultFile is None ):
      resultFile = "out.txt"
      print ("\n!!! No result_file path is provided on the command line!")
      print("Output will be written in default result file- \"{0}\"".format(resultFile))
   else:
      print("\nOutput will be written in - \"{0}\"".format(resultFile))

   timeout_arg = args.t

   'Set default for timeout if no timeout value is provided in the commandline'
   'It is an optional parameter'
   if (timeout_arg is None):
      print ("\n!!! timeout is not on the command line!")
      print ("Default timeout is set as - 60 sec")
      cmd_timeout = 60.0
   else:
      print ("\ntimeout is  - {0} sec".format(timeout_arg))
      cmd_timeout = float(timeout_arg)

   # Register the signal function handler
   signal.signal(signal.SIGALRM, handler)

   # Define a timeout for "runSingleInstance"
   signal.alarm(int(cmd_timeout))
   
   '"runSingleInstance" will continue until any adversarial found or timeout occurs'
   'When timeout occurs codes written within exception will be executed'
   outFile = open(resultFile, "w")
   try:
       # Run for the full timeout period
       retStatus = runSingleInstance(onnxFile, vnnlibFile, cmd_timeout)
       
       # Write results to file
       outFile.write(retStatus)
       print("\nOutput is written in - \"{0}\"".format(resultFile))
   except Exception as exc:
       print(exc)
       outFile.write("timeout," + str(cmd_timeout) + "\n")
       print("\nOutput is written in - \"{0}\"".format(resultFile))
       print("\n!!! Timeout occurred after {0} seconds".format(cmd_timeout))
   
   outFile.close()
