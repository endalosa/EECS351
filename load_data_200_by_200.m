clear;
clc;
close all;

%% load train data from file, rescale to 200x200 and put into 2 matrices: allDogs and allCats
%% then save the .mat file
%% this way they are preclassified for creating our models.
%% beware this code takes a LONG time to run.

%% CATS
allCats = zeros(200,200,3);
for a = 0:12499
   filename = ['cat.' num2str(a,'%2d') '.jpg'];
   img = imread(filename);
   imgResized = imresize(img,[200,200]);
   allCats = cat(4, allCats, imgResized);   
   if a == 1
    allCats = allCats(:,:,:,2);
   end
end

%% DOGS
allDogs = zeros(200,200,3);
for a = 0:12499
   filename = ['dog.' num2str(a,'%2d') '.jpg'];
   img = imread(filename);
   imgResized = imresize(img,[200,200]);
   allDogs = cat(4, allDogs, imgResized);   
   if a == 1
    allDogs = allDogs(:,:,:,2);
   end
end

% filename = 'TRAIN_allcats_alldogs_matrices.mat';
% save(filename)

%%
TEST_CAT = allCats(:,:,:,6250:12499);
TRAIN_CAT = allCats(:,:,:,1:6250);
TEST_DOG = allDogs(:,:,:,6250:12499); 
TRAIN_DOG = allDogs(:,:,:,1:6250);
%%
save("dataOnly.mat",'TEST_DOG', 'TRAIN_DOG', 'TEST_CAT', 'TRAIN_CAT')