import argparse
from images.train_cnn import TrainCnnImageModel
from images.model import ImageModels

# using argparse to parse the argument from command line
parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("-i", help="input folder")
parser.add_argument("-e", help="number of epochs")
parser.add_argument("-s", help="save screenshot")
parser.add_argument("-p", help="early stop")
parser.add_argument("-m", help="monitor")
args: argparse.Namespace = parser.parse_args()

# obtain input directory from command line
inputDir: str = args.i

if args.e is not None:
    TrainCnnImageModel.epochs = int(args.e)
if args.s is not None:
    TrainCnnImageModel.savefig = bool(args.s)
if args.p is not None:
    TrainCnnImageModel.stop_early = bool(args.p)
if args.m is not None:
    TrainCnnImageModel.monitor = str(args.m)

TrainCnnImageModel.train(inputDir, [], "ideal")
TrainCnnImageModel.train(inputDir, [], "greatest_square")
