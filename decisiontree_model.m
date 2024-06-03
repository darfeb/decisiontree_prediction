%Import CSV data
raw_data = 'Covid_Dataset.csv';%covid dataset contain 5434 row and 21 column
open = readtable(raw_data);%read the data
X = open{:,1:20}; %data input = 20 variables
y = open{:,21}; %data output = 1 variable

%Data Preprocessing

%Missing value
data = ismissing(open);%find missing value 
data(1:5,:);

%Outlier 
dataoutlier = isoutlier(X(:,1)); %outlier check
% dataafter = filloutliers(X(:,1),'nearest','mean');%Handling Outlier

%Data Transformation

%min-max normalization
databaru = open;
s = X;

for i = 1:length(s(1,:))
    for j = 1:length(s(:,1))
        databaru{j,i}=(int8(s(j,i))-min(s(:,i)))/(max(s(:,i))-min(s(:,i)));
    end
end
%disp(databaru)

%Split data training 70 % and data testing 30%
databaru = datasample(databaru,5434);
X = databaru{:,1:20};
y = databaru{:,21};
X_train = X(1:3804,:); 
y_train = y(1:3804,:);
X_test = X(3805:5434,:);
y_test = y(3805:5434,:);


%Classification using Decision Tree method
tree = fitctree(X_train,y_train); %Train the data
view(tree)
view(tree,"Mode","graph")

%Evaluation Confusion Matrix
%Predicting X_test
pred_test=predict(tree,X_test);% fit X_test

%Confusion Matrix
cm=confusionmat(y_test,pred_test);

%Accuracy
iscorrect = string(pred_test)==string(y_test);
accuracy = sum(iscorrect)/numel(iscorrect)

fig = figure;
%confusion chart for Overall for Data Training and Testing
confchart=confusionchart(y_test,pred_test);
confchart.Title = 'Confusion Matrix Using Decision Tree for Data Testing';
fig_Position = fig.Position;
fig_Position(2) = fig_Position(2)*0.5;
fig.Position = fig_Position;