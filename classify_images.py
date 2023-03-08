from images.classifier import Classifier

Classifier.classify("./training", "ideal")
Classifier.classify("./training", "greatest_square")
Classifier.classify_utf_face("./training", "ideal")
Classifier.classify_utf_face("./training", "greatest_square")
