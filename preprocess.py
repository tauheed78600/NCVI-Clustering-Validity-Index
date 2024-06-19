import numpy as np
import pandas as pd

# Function to label Chronic Kidney Disease dataset
def Label_ckd(file):
    file = np.array(file)
    label_coln = file[:, 24]
    Label = []
    for i in range(len(label_coln)):
        if label_coln[i] == "ckd":  # ckd(Chronic Kidney Disease)
            Label.append(1)
        else:
            Label.append(0)
    return Label

# Function to label Iris dataset
def Label_iris(file):
    file = np.array(file)
    label_coln = file[:, 4]
    Label = []
    for i in range(len(label_coln)):
        if label_coln[i] == "Iris-setosa":
            Label.append(0)
        elif label_coln[i] == "Iris-versicolor":
            Label.append(1)
        else:  # Iris-virginica
            Label.append(2)
    return Label

# Preprocessing function for Chronic Kidney Disease dataset
def processing_ckd():
    file = pd.read_csv("Dataset/chronic_kidney_disease.csv", header=0)  # Read input data
    label = Label_ckd(file)  # Create label from a input file
    np.savetxt("Dataset_processed/Label.csv", label, delimiter=",", fmt="%d")

    file = file.replace(to_replace="normal", value=0)
    file = file.replace(to_replace="abnormal", value=1)
    file = file.replace(to_replace="present", value=0)
    file = file.replace(to_replace="notpresent", value=1)
    file = file.fillna(0)
    file = file.replace(to_replace="\tyes", value=0)
    file = file.replace(to_replace="yes", value=0)
    file = file.replace(to_replace="no", value=1)
    file = file.replace(to_replace="yes", value=0)
    file = file.replace(to_replace="\tno", value=1)
    file = file.replace(to_replace="good", value=0)
    file = file.replace(to_replace="poor", value=1)
    file = file.replace(to_replace="ckd", value=0)
    file = file.replace(to_replace="ckd\t", value=0)
    file = file.replace(to_replace="notckd", value=1)

    data = np.array(file).astype(float)
    np.savetxt("Dataset_processed/Process.csv", data, delimiter=",", fmt="%f")

    return data, label

# Preprocessing function for Iris dataset
def processing_iris():
    file = pd.read_csv("Dataset/IRIS.csv", header=0)  # Read input data
    label = Label_iris(file)  # Create label from the input file
    np.savetxt("Dataset_processed/Label_iris.csv", label, delimiter=",", fmt="%d")

    # Replace categorical labels with numeric labels
    file = file.replace(to_replace="Iris-setosa", value=0)
    file = file.replace(to_replace="Iris-versicolor", value=1)
    file = file.replace(to_replace="Iris-virginica", value=2)

    data = np.array(file).astype(float)
    np.savetxt("Dataset_processed/Process_iris.csv", data, delimiter=",", fmt="%f")

    return data, label

# Function to preprocess and save both datasets
def processing():
    processing_ckd()
    processing_iris()

def Pre_process():
    processing()
    Data_ckd = pd.read_csv("Dataset_processed/Process_ckd.csv", header=None)
    Label_ckd = pd.read_csv("Dataset_processed/Label_ckd.csv", header=None)
    Data_iris = pd.read_csv("Dataset_processed/Process_iris.csv", header=None)
    Label_iris = pd.read_csv("Dataset_processed/Label_iris.csv", header=None)

    return (Data_ckd, np.array(Label_ckd)), (Data_iris, np.array(Label_iris))

Pre_process()