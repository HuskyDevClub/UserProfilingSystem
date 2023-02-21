import csv
import json
import os
from typing import Any, Optional

from utils.user import User, Users


class TrainAverageModel:
    MODEL_PATH: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model.json"
    )

    @classmethod
    def train(cls, _in: str):
        # database for storing user information, key is user id
        database: dict[str, User] = {}
        # all the information types
        keys: Optional[tuple[str, ...]] = None

        # read data
        with open(os.path.join(_in, "profile", "profile.csv"), newline="") as f:
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

        # statistic variables
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

        # calculate statistic
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

        # generate result
        result: dict = {
            "gender": "male" if total_males >= total_females else "female",
            "age_group": max(age_groups_statistic, key=age_groups_statistic.get),  # type: ignore
        }
        for key in personalities_score_total:
            result[key] = round(personalities_score_total[key] / len(database), 8)

        # save result
        with open(cls.MODEL_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False, sort_keys=True)

        # show result
        print("most common gender:", result["gender"])
        print("age groups statistic:")
        for key in age_groups_statistic:
            print("-", key, ":", age_groups_statistic[key])
        print("personality average:")
        for key in personalities_score_total:
            print(
                "-", key, ":", round(personalities_score_total[key] / len(database), 8)
            )
