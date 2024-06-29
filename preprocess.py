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

# Function to label Zoo dataset
def Label_zoo(file):
    file = np.array(file)
    label_coln = file[:, -1]
    Label = []
    for label in label_coln:
        Label.append(int(label))  # Labels are already numeric in the Zoo dataset
    return Label

# Function to label Spambase dataset
def Label_spambase(file):
    label_coln = file.iloc[:, -1]  # Use DataFrame indexing to get the last column
    Label = label_coln.tolist()  # Labels are already numeric in the Spambase dataset
    return Label

# Preprocessing function for Chronic Kidney Disease dataset
def processing_ckd():
    file = pd.read_csv("Dataset/chronic_kidney_disease.csv", header=0)  # Read input data
    label = Label_ckd(file)  # Create label from input file
    np.savetxt("Dataset_processed/Label_ckd.csv", label, delimiter=",", fmt="%d")

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
    np.savetxt("Dataset_processed/Process_ckd.csv", data, delimiter=",", fmt="%f")

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

# Preprocessing function for Zoo dataset
def processing_zoo():
    file = pd.read_csv("Dataset/zoo.data", header=None)  # Read input data
    label = Label_zoo(file)  # Create label from the input file
    np.savetxt("Dataset_processed/Label_zoo.csv", label, delimiter=",", fmt="%d")

    # Remove the animal name column as it is not needed for processing
    file = file.drop(columns=[0])

    data = np.array(file).astype(float)
    np.savetxt("Dataset_processed/Process_zoo.csv", data, delimiter=",", fmt="%f")

    return data, label

# Preprocessing function for Spambase dataset
def processing_spambase():
    file = pd.read_csv("Dataset/spambase.data", header=None)  # Read input data
    label = Label_spambase(file)  # Create label from the input file
    np.savetxt("Dataset_processed/Label_spambase.csv", label, delimiter=",", fmt="%d")

    data = file.iloc[:, :-1].values  # All columns except the last one as data
    np.savetxt("Dataset_processed/Process_spambase.csv", data, delimiter=",", fmt="%f")

    return data, label

# Function to preprocess and save all datasets
def processing():
    processing_ckd()
    processing_iris()
    processing_zoo()
    processing_spambase()

def Pre_process():
    processing()
    Data_ckd = pd.read_csv("Dataset_processed/Process_ckd.csv", header=None)
    Label_ckd = pd.read_csv("Dataset_processed/Label_ckd.csv", header=None)
    Data_iris = pd.read_csv("Dataset_processed/Process_iris.csv", header=None)
    Label_iris = pd.read_csv("Dataset_processed/Label_iris.csv", header=None)
    Data_zoo = pd.read_csv("Dataset_processed/Process_zoo.csv", header=None)
    Label_zoo = pd.read_csv("Dataset_processed/Label_zoo.csv", header=None)
    Data_spambase = pd.read_csv("Dataset_processed/Process_spambase.csv", header=None)
    Label_spambase = pd.read_csv("Dataset_processed/Label_spambase.csv", header=None)

    return (Data_ckd, np.array(Label_ckd)), (Data_iris, np.array(Label_iris)), (Data_zoo, np.array(Label_zoo)), (Data_spambase, np.array(Label_spambase))

Pre_process()
