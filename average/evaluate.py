import csv
import json
import os
from typing import Any, Optional

from utils.user import User, Users

from .train import TrainAverageModel


class EvaluateAverageModel:
    __MODEL: dict = {}

    with open(TrainAverageModel.MODEL_PATH, "r", encoding="utf-8") as f:
        __MODEL.update(dict(json.load(f)))

    # predict the result
    @classmethod
    def predict(cls, _user: User):
        return User(
            _user.get_id(),
            Users.convert_to_age(cls.__MODEL["age_group"]),
            cls.__MODEL["gender"],
            cls.__MODEL["extrovert"],
            cls.__MODEL["neurotic"],
            cls.__MODEL["agreeable"],
            cls.__MODEL["conscientious"],
            cls.__MODEL["open"],
        )

    # predict the result
    @classmethod
    def process(cls, _input: str, _output: str):
        # all the information types
        keys: Optional[tuple[str, ...]] = None

        # read the csv file
        with open(os.path.join(_input, "profile", "profile.csv"), newline="") as f:
            spamreader = csv.reader(f, delimiter=" ", quotechar="|")

            # process data from csv file
            for row in spamreader:
                _temp: list[str] = row[0].split(",")
                if keys is not None:
                    userInfo: dict[str, Any] = {}
                    # assert len(keys) == len(_temp)
                    for i, _value in enumerate(keys):
                        userInfo[_value] = _temp[i]
                    cls.predict(Users.from_dict(userInfo)).save(_output)
                else:
                    keys = tuple(_temp)
