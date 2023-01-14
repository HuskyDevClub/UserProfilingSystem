import argparse
import csv
import os
from typing import Any, Optional

from user import User

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

# all the information types
keys: Optional[tuple[str, ...]] = None

# read the csv file
with open(os.path.join(inputDir, "profile", "profile.csv"), newline="") as f:
    spamreader = csv.reader(f, delimiter=" ", quotechar="|")

    # process data from csv file
    for row in spamreader:
        _temp: list[str] = row[0].split(",")
        if keys is not None:
            userInfo: dict[str, Any] = {}
            # assert len(keys) == len(_temp)
            for i, _value in enumerate(keys):
                userInfo[_value] = _temp[i]
            User(
                userInfo["userid"],
                "xx-24",
                "female",
                3.48685789,
                2.73242421,
                3.58390421,
                3.44561684,
                3.90869053,
            ).save(outputDir)
        else:
            keys = tuple(_temp)
