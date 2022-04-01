Matlab code for task II KNN
%% make confusion matrix for KNN

%% load .mat file
clear;
clc;
close all;
load("DATA.mat");

%% Combine matrices
%%labels
labels = "cat";
for i=2:800
    labels = cat(2,labels,"cat");
end
for i=1:800
    labels = cat(2,labels,"dog");
end

%%images
test1 = TRAIN_CAT(:,:,:,1);
images = test1(:);

for i=2:800
    test = TRAIN_CAT(:,:,:,i);
    test = test(:);
    images = cat(2,images,test);
end

for i=1:800
    test = TRAIN_DOG(:,:,:,i);
    test = test(:);
    images = cat(2,images,test);
end

%% Create model
Mdl = fitcknn(transpose(images),transpose(labels));

%% Create test data
test1 = TEST_CAT(:,:,:,1);
TESTimages = test1(:);

for i=2:200
    test = TEST_CAT(:,:,:,i);
    test = test(:);
    TESTimages = cat(2,TESTimages,test);
end

for i=1:200
    test = TEST_DOG(:,:,:,i);
    test = test(:);
    TESTimages = cat(2,TESTimages,test);
end

%%labels
correctLabels = "cat";
for i=2:200
    correctLabels = cat(2,correctLabels,"cat");
end
for i=1:200
    correctLabels = cat(2,correctLabels,"dog");
end

%% Run test data thru model
predictedGroups = predict(Mdl, transpose(TESTimages));
C = confusionmat(correctLabels,predictedGroups);
confusionchart(C);
title("KNN Classification Confusion Matrix");
title({'KNN Classification Confusion Matrix','trained on 800 cats & 800 dogs','tested on 200 cats & 200 dogs'}) 
