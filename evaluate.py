import argparse
import os

from average.evaluate import EvaluateAverageModel

# using argparse to parse the argument from command line
parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("-i", help="input folder")
parser.add_argument("-o", help="output folder")
args: argparse.Namespace = parser.parse_args()

# obtain input and output directory from command line
inputDir: str = args.i
outputDir: str = args.o

# check to see if out folder folder exists
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

EvaluateAverageModel.process(inputDir, outputDir)
