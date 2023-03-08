import os
import pickle
import random
import shutil
import joblib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import xml.etree.ElementTree as ET


def text_prediction(input_directory: str, output_directory: str):
    profile_directory = os.path.join("/data", "training", "profile")
    text_directory = os.path.join("/data", "training", "text")
    # tsv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'User_Text_to_Gender.tsv')
    tsv_file = "User_Text_to_Gender.tsv"

    if os.path.exists(tsv_file):
        os.remove(tsv_file)

    with open(tsv_file, "w") as tsv:
        tsv.write("Id" + "\t" + "status" + "\t" + "gender" + "\n")
        with open(os.path.join(profile_directory, "profile.csv")) as data_file:
            next(data_file)

            user_id = ""
            text = ""
            gender = 0.0

            for line in data_file:
                data = line.split(",")
                user_id = data[1]
                with open(
                    os.path.join(text_directory, user_id + ".txt"),
                    "r",
                    encoding="latin1",
                ) as text_file:
                    text = text_file.read().replace("\n", "")

                gender = float(data[3])
                tsv.write(str(user_id) + "\t" + str(text) + "\t" + str(gender) + "\n")

    profile_directory2 = input_directory + "profile"
    text_directory2 = input_directory + "text"
    # tsv_file2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'User_Text_to_Gender2.tsv')
    tsv_file2 = "User_Text_to_Gender2.tsv"

    if os.path.exists(tsv_file2):
        os.remove(tsv_file2)

    with open(tsv_file2, "w") as tsv:
        tsv.write("Id" + "\t" + "status" + "\t" + "gender" + "\n")
        with open(os.path.join(profile_directory2, "profile.csv")) as data_file:
            next(data_file)

            user_id = ""
            text = ""
            gender = 0.0

            for line in data_file:
                data = line.split(",")
                user_id = data[1]
                with open(
                    os.path.join(text_directory, user_id + ".txt"),
                    "r",
                    encoding="latin1",
                ) as text_file:
                    text = text_file.read().replace("\n", "")

                gender = 0
                tsv.write(str(user_id) + "\t" + str(text) + "\t" + str(gender) + "\n")

    with open(tsv_file, "r") as tsv:
        df = pd.read_table(tsv, encoding="ISO-8859-1")

    with open(tsv_file2, "r") as tsv:
        df2 = pd.read_table(tsv, encoding="ISO-8859-1")

    # df_fixed, df2_fixed = df.fillna(' '), df2.fillna(' ')
    # print(df)
    # print(df2)

    data_statuses = df.loc[:, ["status", "gender"]]
    test_data = df2.loc[:, ["status", "gender"]]

    n = 1500

    all_Ids = np.arange(len(data_statuses))

    random.shuffle(all_Ids)
    test_Ids = all_Ids[0:n]
    train_Ids = all_Ids[n:]
    data_test = data_statuses.loc[test_Ids, :]
    data_train = data.statuses.loc[train_Ids, :]

    vect = CountVectorizer()

    X_train = vect.fit_transform(data_train["status"])

    y_train = data_train["gender"]

    clf = LogisticRegression()

    clf.fit(X_train, y_train)

    pickled_model = "finalized_model.sav"
    pickle.dump(clf, open(pickled_model, "wb"))

    X_test2 = vect.transform(test_data["status"])

    y_test = data_test["gender"]

    loaded_model = joblib.load(pickled_model)
    y_predicted = loaded_model.predict(X_test2)

    def getPredictedValue(userID):
        getIndex = df2.loc[df2["Id"] == userID].index.values[0]

        if y_predicted.item(getIndex) == 0:
            return "male"
        else:
            return "female"

    print("\nRunning script...\n\n\n")

    print("Input directory: " + input_directory)
    print("Output directory: " + output_directory)

    def storeGender(gender, genders):
        genders[gender] += 1

    def storeAge(age, ages):
        if age < 25:
            ages[0] += 1
        elif age < 35:
            ages[1] += 1
        elif age < 50:
            ages[2] += 1
        else:
            ages[3] += 1

    def storePersonality(personality, personalities):
        for i in range(0, 5):
            personalities[i] += float(personality[i])

    genders = [0, 0]
    ages = [0, 0, 0, 0]
    personalities = [0, 0, 0, 0, 0]

    age_groups = ["xx-24", "25-34", "35-49", "50-xx"]
    count = 0

    with open(input_directory + "profile/profile.csv") as data_file:
        next(data_file)

        for line in data_file:
            data = line.split(",")

            id = data[1]
            age = 0
            gender = 0

            storeGender(int(gender), genders)
            storeAge(age, ages)

            count += 1

    popular_gender = ""
    popular_age = ""
    popular_personality = [2.5, 2.5, 2.5, 2.5, 2.5]

    if genders[0] > genders[1]:
        popular_gender = "male"
    else:
        popular_gender = "female"

    max_age_count = 0

    for i in range(0, 4):
        if ages[i] > max_age_count:
            max_age_count = ages[i]
            popular_age = age_groups[i]

    # Printing the below for debugging reasons
    print("\n****************************************")
    print("Gender Counts: ")
    print(genders)
    print("Popular Gender: ")
    print(popular_gender)
    print("Age Counts: ")
    print(ages)
    print("Popular Age: ")
    print(popular_age)
    print("Total Personality Sum: ")
    print(personalities)
    print("Total Persons Count: ")
    print(count)
    print("Popular Personality: ")
    print(popular_personality)
    print("****************************************\n")

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)

    os.makedirs(output_directory)

    print("\nWriting results to XML files...\n")

    with open(input_directory + "profile/profile.csv") as data_file:
        next(data_file)

        for line in data_file:
            data = line.split(",")

            id = data[1]

            user = ET.Element("user")
            user.set("id", id)
            user.set("age_group", popular_age)
            user.set("gender", getPredictedValue(id))
            user.set("extrovert", str(popular_personality[0]))
            user.set("neurotic", str(popular_personality[1]))
            user.set("agreeable", str(popular_personality[2]))
            user.set("conscientious", str(popular_personality[3]))
            user.set("open", str(popular_personality[4]))
            user_data = ET.tostring(user)
            user_file = open(output_directory + id + ".xml", "wb")
            user_file.write(user_data)

    print("Finished")
