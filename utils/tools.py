import os
import pandas  # type: ignore

from images.classifier import Classifier

from .user import Users


# classify the images and copy the images file into a specific folder.
def classify(_in: str) -> None:
    Classifier.classify(_in, "ideal")
    Classifier.classify(_in, "greatest_square")
    Classifier.classify_utf_face(_in, "ideal")
    Classifier.classify_utf_face(_in, "greatest_square")


# compare the difference between output csv and target csv
def compare(inDir: str, outDir: str) -> None:
    profile = Users.load_database(os.path.join(inDir, "profile", "profile.csv"))
    out = pandas.read_csv(os.path.join(outDir, "image_out.csv"))

    age_error_counter = 0
    gender_error_counter = 0
    for index, row in out.iterrows():
        if row["age"] != profile[row["userid"]].get_age_group():
            age_error_counter += 1
        if row["gender"] != profile[row["userid"]].get_gender():
            gender_error_counter += 1

    print(age_error_counter)
    print(gender_error_counter)
