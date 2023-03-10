import argparse
import os
import shutil

import pandas  # type: ignore

from images.evaluate import EvaluateImageModel
from images.model import ImageModels
from text.main import text_prediction
from likes.likes import likes_prediction
from utils.user import Users

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

# predict based on text
text_prediction(inputDir, outputDir)
# predict based on image
EvaluateImageModel.process(inputDir, outputDir, "csv")
# predict based on likes
likes_prediction(inputDir, outputDir, _o_type="csv")

resultCsvPath: dict[str, str] = {
    "image": os.path.join(outputDir, "image_out.csv"),
    "text": os.path.join(outputDir, "text_out.csv"),
    "likes": os.path.join(outputDir, "likes_out.csv")
}


resultsInCsv: dict[str, pandas.DataFrame] = {}

for k, v in resultCsvPath.items():
    resultsInCsv[k] = pandas.read_csv(v)
    os.remove(v)

profile: pandas.DataFrame = pandas.read_csv(
    os.path.join(inputDir, "profile", "profile.csv")
)

for index, row in profile.iterrows():
    classification_counter: dict[str, dict[str, int]] = {
        "gender": {"male": 0, "female": 0},
        "age": {"xx-24": 0, "25-34": 0, "35-49": 0, "50-xx": 0},
    }
    lr_counter: dict[str, int] = {}
    for k in ImageModels.OCEAN:
        lr_counter[k[:3]] = 0
    for v in resultsInCsv.values():
        each_vote = v.loc[index]  # type: ignore
        classification_counter["gender"][each_vote["gender"]] += 1
        classification_counter["age"][each_vote["age"]] += 1
        for k in ImageModels.OCEAN:
            classification_counter[k[:3]] += each_vote[k[:3]]

    row["gender"] = max(
        classification_counter["gender"], key=classification_counter["gender"].get  # type: ignore
    )
    row["age"] = max(
        classification_counter["age"], key=classification_counter["age"].get  # type: ignore
    )
    for k in ImageModels.OCEAN:
        row[k] = round(lr_counter[k] / len(resultsInCsv), 3)
    Users.from_dict(row.to_dict()).save(outputDir)
