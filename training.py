import csv
import os
from typing import Any, Optional

from user import User, Users

inputDir: str = "training"
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
            theUser = Users.from_dict(userInfo)
            database[theUser.get_id()] = theUser
        else:
            keys = tuple(_temp)

"""statistic"""
personalities_score_total: dict[str, float] = {
    "extrovert": 0.0,
    "neurotic": 0.0,
    "agreeable": 0.0,
    "conscientious": 0.0,
    "open": 0.0,
}
total_males: int = 0
total_females: int = 0
age_groups_statistic: dict[str, int] = {}

for value in database.values():
    _user_data: dict[str, Any] = value.to_dict()
    for key in personalities_score_total:
        personalities_score_total[key] += _user_data[key]
    if value.get_gender() == "male":
        total_males += 1
    else:
        total_females += 1
    age_groups_statistic[value.get_age_group()] = (
        age_groups_statistic.get(value.get_age_group(), 0) + 1
    )

print("most common gender:", "male" if total_males >= total_females else "female")
print("age groups statistic:")
for key in age_groups_statistic:
    print("-", key, ":", age_groups_statistic[key])
print("personality average:")
for key in personalities_score_total:
    print("-", key, ":", round(personalities_score_total[key] / len(database), 8))
