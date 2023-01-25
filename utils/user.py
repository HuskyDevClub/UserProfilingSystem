import os
import xml.etree.cElementTree as ET
from typing import Any


class User:
    def __init__(
        self,
        _id: str,
        _age_group: str,
        _gender: str,
        _extrovert: float | int,
        _neurotic: float | int,
        _agreeable: float | int,
        _conscientious: float | int,
        _open: float | int,
    ) -> None:
        self.__id: str = _id
        self.__age_group: str = _age_group
        self.__gender: str = _gender
        self.__extrovert: float | int = _extrovert
        self.__neurotic: float | int = _neurotic
        self.__agreeable: float | int = _agreeable
        self.__conscientious: float | int = _conscientious
        self.__open: float | int = _open

    def get_id(self) -> str:
        return self.__id

    def get_age_group(self) -> str:
        return self.__age_group

    def get_gender(self) -> str:
        return self.__gender

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.__id,
            "age_group": self.__age_group,
            "gender": self.__gender,
            "extrovert": self.__extrovert,
            "neurotic": self.__neurotic,
            "agreeable": self.__agreeable,
            "conscientious": self.__conscientious,
            "open": self.__open,
        }

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
                    'age_group="{}"\n'.format(self.__age_group),
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
    def convert_age_group(_age: float | int) -> str:
        _theAge: float = float(_age)
        if _theAge <= 24:
            return "xx-24"
        elif _theAge <= 34:
            return "25-34"
        elif _theAge <= 49:
            return "35-49"
        else:
            return "50-xx"

    @staticmethod
    def convert_gender(_gender: int) -> str:
        return (
            "male"
            if _gender == "0.0"
            else "female"
            if _gender == "1.0"
            else Exception("unknown gender")
        )

    @classmethod
    def from_dict(cls, _data: dict[str, Any]) -> User:
        return User(
            _data["userid"],
            cls.convert_age_group(_data["age"]),
            cls.convert_gender(_data["gender"]),
            float(_data["ext"]),
            float(_data["neu"]),
            float(_data["agr"]),
            float(_data["con"]),
            float(_data["ope"]),
        )
