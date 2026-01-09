%% 
% Logistic Regression vs. Artificial Neural Network for Prediction of Heart Disease 
% Chase Roby
% 10/17/2025

data = readtable('heart.csv');

fprintf('Dataset size: %d rows, %d columns\n', height(data), width(data));
disp('Column names:'); disp(data.Properties.VariableNames);

summary(data)

% Sum of missing values - none shown
sum(ismissing(data))


% Figure 1 - Exploratory analysis of Distribution of patient ages
figure;
histogram(data.Age, 15);
xlabel('Age (years)');
ylabel('Number of Patients');
title('Fig 1. Distribution of Patient Ages');
grid on;

% Figure 2 - Exploratory analysis of class balance
figure;
diseaseCounts = groupcounts(data.HeartDisease);
bar(categorical({'No Heart Disease','Heart Disease'}), diseaseCounts);
ylabel('Number of Patients');
title('Fig 2. Heart Disease vs. No Heart Disease');
grid on;


% Convert Categorical Predictors

data.Sex = categorical(data.Sex);
data.ChestPainType = categorical(data.ChestPainType);
data.RestingECG = categorical(data.RestingECG);
data.ExerciseAngina = categorical(data.ExerciseAngina);
data.ST_Slope = categorical(data.ST_Slope);

% One-Hot Encoding of the Categorical Variables
encodedData = data;
categoricalVars = varfun(@iscategorical, data, 'OutputFormat', 'uniform');
categoricalNames = data.Properties.VariableNames(categoricalVars);

for i = 1:length(categoricalNames)
    
    varName = categoricalNames{i};
    catVar = data.(varName);
    
    % Get dummy variables
    dummies = dummyvar(catVar);
    dummyCategories = categories(catVar);
    
    % Add dummy columns to encodedData
    for j = 1:size(dummies, 2)
        newColName = [varName '_' char(dummyCategories{j})];
        encodedData.(newColName) = dummies(:, j);
    end
    
    % Remove original categorical variable
    encodedData.(varName) = [];
end

% Standardize Continuous Predictors

continuous = ["Age","RestingBP","Cholesterol","MaxHR","Oldpeak"];

for i = 1:length(continuous)
    col = encodedData.(continuous(i));
    encodedData.(continuous(i)) = (col - mean(col)) / std(col);
end

% Split into Predictors and Labels 

X = encodedData(:, encodedData.Properties.VariableNames ~= "HeartDisease");
Y = encodedData.HeartDisease;
Y = double(Y);

% Creating Unified 70/15/15 splits

N = height(X);   % number of samples
idx = randperm(N);   % shuffle indices

% Compute split points
nTrain = round(0.70 * N);
nVal   = round(0.15 * N);
nTest  = N - nTrain - nVal;   % ensures all samples are used

% Assign indices
trainIdx = idx(1:nTrain);
valIdx   = idx(nTrain+1 : nTrain+nVal);
testIdx  = idx(nTrain+nVal+1 : end);

% Create the splits
Xtrain = X(trainIdx, :);
Ytrain = Y(trainIdx, :);

Xval = X(valIdx, :);
Yval = Y(valIdx, :);

Xtest = X(testIdx, :);
Ytest = Y(testIdx, :);

% Train Logistic Regression Model

model_lr = fitglm(Xtrain, Ytrain, "Distribution", "binomial");


% Predict on test set

probs_lr = predict(model_lr, Xtest);
preds_lr = probs_lr > 0.5;

% Compute Performance metrics

preds_lr = double(preds_lr);
cm_lr = confusionmat(Ytest, preds_lr);

accuracy_lr = mean(preds_lr == Ytest);

precision_lr = cm_lr(2,2) / (cm_lr(2,2) + cm_lr(1,2));
recall_lr = cm_lr(2,2) / (cm_lr(2,2) + cm_lr(2,1));
f1_lr = 2 * (precision_lr * recall_lr) / (precision_lr + recall_lr);

% Train the ANN - Conversion to Matrices
Xtrain_ann = Xtrain{:,:}';
Ytrain_ann = Ytrain';

Xval_ann = Xval{:,:}';
Yval_ann = Yval';

Xtest_ann = Xtest{:,:}';
Ytest_ann = Ytest';


% Combining for Cross-Validation

Xall = [Xtrain_ann Xval_ann];  
Yall = [Ytrain_ann Yval_ann];  

% K-fold Validation

% Architecture candidates being tested
% Some multi-layer variations included in order to test extra depth and
% and complexity 

architectures = {[10], [20], [40], [60], [100], [20 10], [40 10], [60 30], [40 20 10]};

% Number of folds
% Tried 10, took forever to run 

K = 5;

% The Cross Validation (cv) partition
cv = cvpartition(length(Yall), 'KFold', K);

% Store accuracies 

cvAccuracy = zeros(length(architectures),K);
cvPrecision = zeros(length(architectures),K);
cvRecall = zeros(length(architectures),K);
cvF1 = zeros(length(architectures),K);

% Looping through each candidate
for h = 1:length(architectures)
    layers = architectures{h};
    fprintf('Testing Architecture: [%s]\n', num2str(layers));
    
    for fold = 1:K % Looping through each fold
        trainIdx = training(cv, fold);
        valIdx   = test(cv, fold);

        XtrainCV = Xall(:, trainIdx);
        YtrainCV = Yall(:, trainIdx);
        XvalCV   = Xall(:, valIdx);
        YvalCV   = Yall(:, valIdx);

        % Creating and training neural network for validation

        net = patternnet(layers);
        net.trainParam.showWindow = false; %% STOPS 50 TABS OPENING!!! 
        net = train(net, XtrainCV, YtrainCV);

        % Validate / Compute Accuracy
        preds = net(XvalCV) > 0.5;
        preds = double(preds);  

        cvAccuracy(h, fold) = mean(double(preds) == YvalCV);

        % Definining extra metrics
        TP = sum(preds & YvalCV);  % True Positives
        FP = sum(preds & ~YvalCV); % False Positives
        FN = sum(~preds & YvalCV); % False Negatives
        
        % Computing Recall, Precision, F1 
        cvPrecision(h, fold) = TP / (TP + FP);
        cvRecall(h, fold)    = TP / (TP + FN);
        cvF1(h, fold)        = 2 * (cvPrecision(h, fold) * cvRecall(h, fold)) / ...
                                (cvPrecision(h, fold) + cvRecall(h, fold));

       
    end
end

% Mean Metrics
meanAcc   = mean(cvAccuracy, 2);
meanPrec  = mean(cvPrecision, 2);
meanRec   = mean(cvRecall, 2);
meanF1    = mean(cvF1, 2);

% Convert architectures to strings for table
archStrings = strings(length(architectures),1);
for h = 1:length(architectures)
    archStrings(h) = mat2str(architectures{h});
end

% Create table with all metrics
resultsCV = table(archStrings, meanAcc, meanPrec, meanRec, meanF1, ...
                  'VariableNames', {'Architecture','Mean Accuracy','Mean Precision','Mean Recall','Mean F1'});

% Display the table
disp('Cross Validation Results:');
disp(resultsCV);

% Create and train a simple MLP

% Single ANN training commented out for architecture testing here; 
% 
% hiddenLayerSize = 40;
% net = patternnet(hiddenLayerSize);
% 
% Tuning Changes
% Potentially epochs / changing learning rate 
% 
% net.divideParam.trainRatio = 0.70;
% net.divideParam.valRatio = 0.15;
% net.divideParam.testRatio = 0.15;
% 
% net = train(net, Xtrain_ann, Ytrain_ann);


