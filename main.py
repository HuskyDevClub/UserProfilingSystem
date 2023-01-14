import csv
import os
from typing import Any, Optional

from user import User

inputDir: str = "public-test-data"
outputDir: str = "output"

# check to see if out folder folder exists
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

# database for storing user information, key is user id
database: dict[str, User] = {}
# all the information types
keys: Optional[tuple[str, ...]] = None

with open(os.path.join(inputDir, "profile", "profile.csv"), newline="") as f:
    # read the csv file
    spamreader = csv.reader(f, delimiter=" ", quotechar="|")
    for row in spamreader:
        _temp: list[str] = row[0].split(",")
        if keys is not None:
            userInfo: dict[str, Any] = {}
            assert len(keys) == len(_temp)
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
