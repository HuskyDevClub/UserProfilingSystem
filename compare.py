from utils.user import Users
import pandas  # type: ignore

profile = Users.load_database("./training/profile/profile.csv")
out = pandas.read_csv("./output/image_out.csv")

age_error_counter = 0
gender_error_counter = 0
for index, row in out.iterrows():
    if row["age"] != profile[row["userid"]].get_age_group():
        age_error_counter += 1
    if row["gender"] != profile[row["userid"]].get_gender():
        gender_error_counter += 1

print(age_error_counter)
print(gender_error_counter)
