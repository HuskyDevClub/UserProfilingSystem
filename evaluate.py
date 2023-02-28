import argparse
import os
import shutil

#from average.evaluate import EvaluateAverageModel
from images.evaluate import EvaluateImageModel

# using argparse to parse the argument from command line
parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("-i", help="input folder")
parser.add_argument("-o", help="output folder")
args: argparse.Namespace = parser.parse_args()

# obtain input and output directory from command line
inputDir: str = args.i
outputDir: str = args.o

# check to see if out folder folder exists
if os.path.exists(outputDir):
    shutil.rmtree(outputDir)
os.mkdir(outputDir)

# EvaluateAverageModel.process(inputDir, outputDir)
EvaluateImageModel.process(inputDir, outputDir)
