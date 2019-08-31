# Hackathon---Loan-default-classification
### Identify if a borrower would default(or not) based on given features

## This is a supervised binary classification problem with an imbalanced dataset.
### Best training and testing prediction accuracy(F1) is found using ExtraTrees Classifier.

## Instructions to execute the notebook:

1. Ensure that both the Notebook(.ipnyb file) and the ProcessClass.py helper class file are located in the same folder
2. Ensure that all input data file(training, testing etc.) are also placed in the same folder as the main notebook
3. All python packages used in the notebook are being imported in the very first code block. If any package is missing, please install the same using conda install or pip install as applicable
4. All the code cells can be run at once in sequence using Run All option. Ensure to monitor the cells to see if all them complete in the right order. There is a number at the extreme left which should be in increasing order 
5. Final prediction file is written out to the same location as original notebook

## Special Notes on execution:

->> Several other methods were tried but not used in the final implementation:
     a) A few new features were tried out with a combination of existing ones. These have been commented in the code
     b) A genetic algorithm was tried for feature reduction/selection but was discarded
     c) A neural network was tried but it overfitted and was hence discarded
     d) Only the best performing classifier - ExtraTrees is executed in the final version. Rest are commented but can be tried out
        if required. The code is designed to retain the classifer with the best training score and use it for prediction
