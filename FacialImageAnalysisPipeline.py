import MachineLearningLibrary

workingFolder = "./demo_results"

maxNumberOfFeatures = 5 # Maximum number of features to test during cross validation

# Original data paths containing images and landmarks
negativeClassImageFolder = "./demo_data/youngerSCD"
positiveClassImageFolder = "./demo_data/olderSCD"

# Working directory: intermediate and final results will be saved under this path
print('Processing at: ' + workingFolder)

# Creating the mGeneML object
mGeneObject = MachineLearningLibrary.ML()
mGeneObject.negativeClassName = 'youngerSCD' # Class label 0 (e.g., control)
mGeneObject.positiveClassName = 'olderSCD' # Class label 1 (e.g., sickle cell or syndromic)
mGeneObject.optimizeThreshold = False
mGeneObject.threshold = 0.5
mGeneObject.asymmetry = False
mGeneObject.featureSelector = 'RecursiveElimination'#'RecursiveElimination' # MCFS_p #SequentialSelection #SelectKBest

# Creates a folder structure, standardize all images and calculates the local binary patterns
print('  - Initializing the folder structure')
mGeneObject.Initialize(negativeClassImageFolder, negativeClassImageFolder,
                     positiveClassImageFolder, positiveClassImageFolder,
                          workingFolder)

# Trains the LDA matrices for cross validation and calculates the all the features
print('  - Calculating features for cross validation')
mGeneObject.CalculateFeaturesForCrossValidation(workingFolder)

# Cross validates the model using the previously calculated featuress
print('  - Cross validating')
mGeneObject.CrossValidate(workingFolder, maxNumberOfFeatures=maxNumberOfFeatures)

# Exports the results to an Excel file
print('  - Exporting cross validation to Excel file')
mGeneObject.ExportCrossValidationToExcel(workingFolder)
