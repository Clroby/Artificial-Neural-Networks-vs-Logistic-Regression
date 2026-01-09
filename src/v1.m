%% 
% Logistic Regression vs. Artificial Neural Network for Prediction of Heart Disease 
% Chase Roby
% 10/17/2025

data = readtable('heart.csv');

fprintf('Dataset size: %d rows, %d columns\n', height(data), width(data));
disp('Column names:'); disp(data.Properties.VariableNames);

summary(data)

%Sum of missing values - none shown
sum(ismissing(data))


%Figure 1 - Exploratory analysis of Distribution of patient ages
figure;
histogram(data.Age, 15);
xlabel('Age (years)');
ylabel('Number of Patients');
title('Fig 1. Distribution of Patient Ages');
grid on;

%Figure 2 - Exploratory analysis of class balance
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
    
    % Add dummy columns to encoded data
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

cv1 = cvpartition(Y, "Holdout", 0.30);

Xtrain = X(training(cv1), :);
Ytrain = Y(training(cv1));

Xtemp = X(test(cv1), :);
Ytemp = Y(test(cv1));

cv2 = cvpartition(Ytemp, "Holdout", 0.50);

Xval = Xtemp(training(cv2), :);
Yval = Ytemp(training(cv2));

Xtest = Xtemp(test(cv2), :);
Ytest = Ytemp(test(cv2));

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

% Create and train a simple MLP

hiddenLayerSize = [20 10];
net = patternnet(hiddenLayerSize);

net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

net = train(net, Xtrain_ann, Ytrain_ann);

% Predict on the test set

probs_ann = net(Xtest_ann);
preds_ann = probs_ann > 0.5;

% Compute ANN Performance 

preds_ann = preds_ann';     
preds_ann = double(preds_ann);
cm_ann = confusionmat(Ytest, preds_ann');

accuracy_lr = sum(preds_lr == Ytest) / numel(Ytest); 
accuracy_ann = sum(preds_ann == Ytest) / numel(Ytest); 

% For precision, recall, F1 -> getting a single value :
precision_lr = sum(preds_lr & Ytest) / sum(preds_lr);  
recall_lr = sum(preds_lr & Ytest) / sum(Ytest);  
f1_lr = 2 * (precision_lr * recall_lr) / (precision_lr + recall_lr);  

precision_ann = sum(preds_ann & Ytest) / sum(preds_ann);
recall_ann = sum(preds_ann & Ytest) / sum(Ytest);
f1_ann = 2 * (precision_ann * recall_ann) / (precision_ann + recall_ann);


% Compare Results 

results = [accuracy_lr, accuracy_ann];
results = table(["LogReg";"ANN"], ...
                [accuracy_lr; accuracy_ann], ...
                [precision_lr; precision_ann], ...
                [recall_lr; recall_ann], ...
                [f1_lr; f1_ann], ...
                'VariableNames', ...
                {'Model','Accuracy','Precision','Recall','F1'});
disp(results)



% Confusion Matrix for Logistic Regression 
figure;
confusionchart(cm_lr, {'No Heart Disease','Heart Disease'});
title('Logistic Regression Confusion Matrix');



% Confusion Matrix for ANN 
figure;
confusionchart(cm_ann, {'No Heart Disease','Heart Disease'});
title('ANN Confusion Matrix');
