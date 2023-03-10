import csv
import os
import xml.etree.cElementTree as ET
from typing import Any, Optional

import pandas  # type: ignore


class User:
    def __init__(
        self,
        _id: str,
        _age: float | int,
        _gender: str,
        _extrovert: float | int,
        _neurotic: float | int,
        _agreeable: float | int,
        _conscientious: float | int,
        _open: float | int,
    ) -> None:
        self.__id: str = _id
        self.__age: int = int(_age)
        self.__gender: str = _gender
        self.__extrovert: float | int = _extrovert
        self.__neurotic: float | int = _neurotic
        self.__agreeable: float | int = _agreeable
        self.__conscientious: float | int = _conscientious
        self.__open: float | int = _open

    def get_id(self) -> str:
        return self.__id

    def get_age(self) -> int:
        return self.__age

    def get_age_group(self) -> str:
        return Users.convert_age_group(self.__age)

    def get_binary_age_group(self) -> str:
        return "xx-24" if self.__age <= 24 else "25-34"

    def get_age_group_index(self) -> int:
        return Users.convert_age_group_index(self.__age)

    def get_gender(self) -> str:
        return self.__gender

    def get_extrovert(self) -> float | int:
        return self.__extrovert

    def get_neurotic(self) -> float | int:
        return self.__neurotic

    def get_agreeable(self) -> float | int:
        return self.__agreeable

    def get_conscientious(self) -> float | int:
        return self.__conscientious

    def get_open(self) -> float | int:
        return self.__open

    def get_ocean(self, key: str) -> float | int:
        if key == "extrovert":
            return self.__extrovert
        elif key == "neurotic":
            return self.__neurotic
        elif key == "agreeable":
            return self.__agreeable
        elif key == "conscientious":
            return self.__conscientious
        elif key == "open":
            return self.__open
        else:
            raise Exception("unknown ocean type")

    def update_ocean_score(self, results: dict[str, float | int]) -> None:
        self.__extrovert = results["extrovert"]
        self.__neurotic = results["neurotic"]
        self.__agreeable = results["agreeable"]
        self.__conscientious = results["conscientious"]
        self.__open = results["open"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.__id,
            "age_group": self.get_age_group(),
            "gender": self.__gender,
            "extrovert": self.__extrovert,
            "neurotic": self.__neurotic,
            "agreeable": self.__agreeable,
            "conscientious": self.__conscientious,
            "open": self.__open,
        }

    def get_data_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame(
            {
                "userid": [self.get_id()],
                "age": [self.get_age_group()],
                "gender": [self.get_gender()],
                "ope": [self.get_open()],
                "con": [self.get_conscientious()],
                "ext": [self.get_extrovert()],
                "agr": [self.get_agreeable()],
                "neu": [self.get_neurotic()],
            }
        )

    def save(self, output_dir: str) -> None:
        _data: dict[str, Any] = self.to_dict()
        for key in _data:
            _data[key] = str(_data[key])
        ET.ElementTree(ET.Element("user", attrib=_data)).write(
            os.path.join(output_dir, "{}.xml".format(self.__id))
        )
        """
        with open(
            os.path.join(output_dir, "{}.xml".format(self.__id)), mode="w"
        ) as f:
            f.writelines(
                [
                    "<user\n",
                    '   id="{}"\n'.format(self.__id),
                    'age_group="{}"\n'.format(self.get_age_group()),
                    'gender="{}"\n'.format(self.__gender),
                    'extrovert="{}"\n'.format(self.__extrovert),
                    'neurotic="{}"\n'.format(self.__neurotic),
                    'agreeable="{}"\n'.format(self.__agreeable),
                    'conscientious="{}"\n'.format(self.__conscientious),
                    'open="{}"\n'.format(self.__open),
                    "/>\n",
                ]
            )
        """


class Users:
    @staticmethod
    def convert_to_age(_age_group: str) -> int:
        if _age_group == "xx-24":
            return 24
        elif _age_group == "25-34":
            return 34
        elif _age_group == "35-49":
            return 49
        elif _age_group == "50-xx":
            return 50
        else:
            raise Exception("Unknown age group:", _age_group)

    @staticmethod
    def convert_age_group(_age: float | int) -> str:
        _theAge: int = round(float(_age))
        if _theAge <= 24:
            return "xx-24"
        elif _theAge <= 34:
            return "25-34"
        elif _theAge <= 49:
            return "35-49"
        else:
            return "50-xx"

    @staticmethod
    def convert_age_group_index(_age: float | int) -> int:
        _theAge: int = round(float(_age))
        if _theAge <= 24:
            return 0
        elif _theAge <= 34:
            return 1
        elif _theAge <= 49:
            return 2
        else:
            return 3

    @staticmethod
    def convert_gender(_gender: str) -> str:
        if _gender == "0.0":
            return "male"
        elif _gender == "1.0":
            return "female"
        elif _gender == "male":
            return "male"
        elif _gender == "female":
            return "female"
        else:
            raise Exception("unknown gender")

    @classmethod
    def from_dict(cls, _data: dict[str, Any]) -> User:
        _age: int = -1
        if (
            not isinstance(_data["age"], str)
            or _data["age"].replace(".", "", 1).isnumeric()
        ):
            _age = round(float(_data["age"]))
        elif _data["age"] != "-":
            _age = cls.convert_to_age(_data["age"])
        return User(
            _data["userid"],
            _age,
            cls.convert_gender(_data["gender"]) if _data["gender"] != "-" else "male",
            float(_data["ext"]) if _data["ext"] != "-" else -1,
            float(_data["neu"]) if _data["neu"] != "-" else -1,
            float(_data["agr"]) if _data["agr"] != "-" else -1,
            float(_data["con"]) if _data["con"] != "-" else -1,
            float(_data["ope"]) if _data["ope"] != "-" else -1,
        )

    @staticmethod
    def load_database(profile_csv_location: str) -> dict[str, User]:
        # database for storing user information, key is user id
        database: dict[str, User] = {}
        # all the information types
        keys: Optional[tuple[str, ...]] = None
        # open csv and start loading data
        with open(profile_csv_location, newline="") as f:
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
        # return data
        return database

    @staticmethod
    def get_profile(profile_csv_location: str) -> pandas.DataFrame:
        return pandas.read_csv(profile_csv_location)


class UserTests:
    @staticmethod
    def run() -> None:
        testUser1: User = User("testId", 24, "male", 1.5, 0.0, 1.0, 5.0, 3.0)
        assert testUser1.get_id() == "testId"
        assert testUser1.get_age_group() == "xx-24"
        assert testUser1.get_extrovert() == 1.5
        assert testUser1.get_neurotic() == 0.0
        assert testUser1.get_agreeable() == 1.0
        assert testUser1.get_conscientious() == 5.0
        assert testUser1.get_open() == 3.0
        assert testUser1.get_gender() == "male"
        testUser2: User = User("testId", 50, "female", 1.5, 0.0, 0.0, 5.0, 3.0)
        assert testUser2.get_id() == "testId"
        assert testUser2.get_age_group() == "50-xx"
        assert testUser2.get_extrovert() == 1.5
        assert testUser2.get_neurotic() == 0.0
        assert testUser2.get_agreeable() == 0.0
        assert testUser2.get_conscientious() == 5.0
        assert testUser2.get_open() == 3.0
        assert testUser2.get_gender() == "female"
        assert Users.convert_age_group(24.0) == "xx-24"


UserTests.run()
