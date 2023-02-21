import argparse

# from average.train import TrainAverageModel
from images.train_cnn import TrainCnnImageModel

# from images.train_dt import TrainImageDecisionTree

# using argparse to parse the argument from command line
parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("-i", help="input folder")
parser.add_argument("-e", help="number of epochs")
parser.add_argument("-s", help="screenshot")
parser.add_argument("-x", help="screenshot")
args: argparse.Namespace = parser.parse_args()

# obtain input directory from command line
inputDir: str = args.i

if args.e is not None:
    TrainCnnImageModel.epochs = int(args.e)
if args.s is not None:
    TrainCnnImageModel.savefig = bool(args.s)

# TrainAverageModel.train(inputDir)
TrainCnnImageModel.train(inputDir, [], int(args.x) if args.x is not None else None)
# TrainImageDecisionTree.train(inputDir, int(args.x) if args.x is not None else None)
