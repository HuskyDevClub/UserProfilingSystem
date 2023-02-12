import argparse

from average.train import TrainAverageModel
from images.train import TrainImageModel

# using argparse to parse the argument from command line
parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("-i", help="input folder")
args: argparse.Namespace = parser.parse_args()

# obtain input directory from command line
inputDir: str = args.i

# TrainAverageModel.train(inputDir)

TrainImageModel.train(inputDir, "g", 5)
