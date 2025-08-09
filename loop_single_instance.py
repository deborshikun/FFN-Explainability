import os

single = "run_single_instance.py"
# onnx_path = "benchmarks/acasxu/ACASXU_run2a_2_9_batch_2000.onnx"
# prop_path = "benchmarks/acasxu/prop_1.vnnlib"

onnx_name = input("Model Name: ")
prop_name = input("Property Name: ")

onnx_path = "benchmarks/acasxu/" + onnx_name + "_batch_2000.onnx"
prop_path = "benchmarks/acasxu/" + prop_name + ".vnnlib"

timeoutval = input("Enter timeout value in seconds (default is 300): ")
if not timeoutval:
    timeoutval = 300

i = 0
while i < 2:
    os.system(f"python3 {single} -m {onnx_path} -p {prop_path} -o {onnx_name}__{prop_name}__Loop{i}.txt -t {timeoutval}")
    i += 1