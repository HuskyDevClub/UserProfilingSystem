import argparse as ap
import os
import numpy as np
import pandas as pd
import time as t

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


def likes_prediction(
        input_directory: str,
        output_directory: str,
        ensemble: bool = True,
        debug: bool = False,
        _o_type: str = "xml",
):
    start_time: float = t.time()

    # Get information
    if debug:
        inputDir: str = "C:/temp/tcss455/public-test-data/"
        outputDir: str = "C:/temp/tcss455/output/"

        df1 = pd.read_csv("C:/temp/tcss455/training/relation/relation.csv", index_col=0)
        df2 = pd.read_csv("C:/temp/tcss455/training/profile/profile.csv", index_col=0)
    else:
        inputDir: str = input_directory
        outputDir: str = output_directory

        df1 = pd.read_csv("/data/training/relation/relation.csv", index_col=0)
        df2 = pd.read_csv("/data/training/profile/profile.csv", index_col=0)

    df3 = pd.read_csv(
        os.path.join(inputDir, "relation", "relation.csv"), index_col=0
    )

    # Filter out the least common likes (occurrence of 1)
    """
    least_common = []
    for like_id, cnt in df1['like_id'].value_counts(ascending=True).iteritems():
        if cnt == 1:
            least_common.append(like_id)
        else:
            break
    df1 = df1[~df1['like_id'].isin(least_common)]
    if debug:
        df1.to_csv('filtered.csv')
    """

    # Filter out the most common likes
    """
    most_common = []
    i = 0
    for like_id, cnt in df1['like_id'].value_counts().iteritems():
        if i < 50:
            most_common.append(like_id)
            i += 1
        else:
            break
    df1 = df1[~df1['like_id'].isin(most_common)]
    if debug:
        df1.to_csv('filtered.csv')
    """

    # Get transcript of likes
    df1 = (
        df1.groupby("userid")["like_id"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )
    df3 = (
        df3.groupby("userid")["like_id"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )

    # Merge the two training data frames together using userid as common element
    dfm = pd.merge(df1, df2, on="userid", how="inner")

    # Convert age into buckets
    dfm["age"] = pd.cut(
        x=dfm["age"],
        bins=[0, 24, 34, 49, 150],
        labels=["xx-24", "25-34", "35-49", "50-xx"],
    )

    # Convert gender 0 and 1 to male and female respectively
    map_to_text = {0: "male", 1: "female"}
    dfm["gender"] = dfm["gender"].replace(map_to_text)

    if debug:
        dfm.to_csv("merged.csv")
        testSize: float = 0.2
    else:
        testSize: int = 1

    count_vect = CountVectorizer()

    if ensemble:
        print("BEGINNING 'LIKES' PROCESS")
    else:
        print("BEGINNING PROCESS")

    # For age and gender
    for column in dfm.columns[2:4]:
        if ensemble:
            print("'Likes' Processing: %s ..." % column[:3])
        else:
            print("Processing: %s ..." % column[:3])
        X_train, X_test, y_train, y_test = train_test_split(
            dfm["like_id"], dfm[column], test_size=testSize
        )
        X_train = count_vect.fit_transform(X_train)
        if column == "age":
            logreg = LogisticRegression(C=1, solver="lbfgs", multi_class="multinomial", max_iter=400)
        else:
            logreg = LogisticRegression()
        logreg.fit(X_train, y_train)

        if debug:
            # Testing logistic regression
            X_test = count_vect.transform(X_test)
            y_predicted = logreg.predict(X_test)

            # Reporting on log reg performance
            print("Predicting", column)
            print("Accuracy: %.2f" % accuracy_score(y_test, y_predicted))
        else:
            df3[column] = logreg.predict(count_vect.transform(df3["like_id"]))

    # For OCEAN
    for column in dfm.columns[4:]:
        if ensemble:
            print("'Likes' Processing: %s ..." % column)
        else:
            print("Processing: %s ..." % column)
        X_train, X_test, y_train, y_test = train_test_split(
            dfm["like_id"], dfm[column], test_size=testSize
        )
        X_train = count_vect.fit_transform(X_train)
        svreg = SVR()
        svreg.fit(X_train, y_train)

        if debug:
            # Testing support vector regression
            X_test = count_vect.transform(X_test)
            y_predicted = svreg.predict(X_test)

            # Reporting on sv reg performance
            print("Predicting", column)
            print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))
        else:
            df3[column] = svreg.predict(count_vect.transform(df3["like_id"]))

    if not debug:
        # Check to see if out folder exists
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        if _o_type == "xml":
            output: str = '<user\n\tid="{}"\nage_group="{}"\ngender="{}"\nextrovert="{}"' \
                          '\nneurotic="{}"\nagreeable="{}"\nconscientious="{}"\nopen="{}"\n/>'
            # Each row to a separate xml
            for row in df3.itertuples():
                with open(os.path.join(outputDir, f"{row[1]}.xml"), "w") as f:
                    f.write(
                        output.format(
                            row[1],  # userid
                            row[3],  # age_group
                            row[4],  # gender
                            row[7],  # ext
                            row[9],  # neu
                            row[8],  # agr
                            row[6],  # con
                            row[5],  # ope
                        )
                    )
        elif _o_type == "csv":
            df3 = df3.drop(["like_id"], axis=1)
            df3.to_csv(os.path.join(outputDir, "likes_out.csv"))

    elapsed_time: float = t.time() - start_time
    if ensemble:
        print("'LIKES' PROCESS COMPLETE: %.2f SECONDS" % elapsed_time)
    else:
        print("PROCESS COMPLETE: %.2f SECONDS" % elapsed_time)


if __name__ == "__main__":
    # Using argparse to parse the argument from command line
    parser: ap.ArgumentParser = ap.ArgumentParser()
    parser.add_argument("-i", help="input folder")
    parser.add_argument("-o", help="output folder")
    parser.add_argument("-d", "--debug", action="store_true", help="debugging help")
    args: ap.Namespace = parser.parse_args()
    likes_prediction(args.i, args.o, False, args.debug)
