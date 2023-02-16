import argparse

from average.train import TrainAverageModel
from images.train import TrainImageModel

# using argparse to parse the argument from command line
parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("-i", help="input folder")
parser.add_argument("-e", help="number of epochs")
parser.add_argument("-s", help="screenshot")
args: argparse.Namespace = parser.parse_args()

# obtain input directory from command line
inputDir: str = args.i
if args.e is not None:
    TrainImageModel.epochs = int(args.e)
if args.s is not None:
    TrainImageModel.savefig = bool(args.s)

# TrainAverageModel.train(inputDir)

TrainImageModel.train(inputDir)
