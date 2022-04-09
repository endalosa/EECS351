load("80_20_DATA.mat");

%% Save to folders
%%TRAIN CATS
folder = sprintf('cats');
for i = 1:10000
    baseFileName = sprintf('train_cat_ #%d.png', i);
    fullFileName = fullfile(folder, baseFileName);
    imwrite(TRAIN_CAT(:,:,:,i), fullFileName);
end

%% Save to folders
%%TRAIN CATS
folder = sprintf('cats');
for i = 1:10000
    baseFileName = sprintf('train_cat_ #%d.png', i);
    fullFileName = fullfile(folder, baseFileName);
    imwrite(TRAIN_CAT(:,:,:,i), fullFileName);
end

%% TEST CATS
folder = sprintf('TESTCAT');
for i = 1:2500
    baseFileName = sprintf('test_cat_ #%d.png', i);
    fullFileName = fullfile(folder, baseFileName);
    imwrite(TEST_CAT_NEW(:,:,:,i), fullFileName);
end


%% TEST DOG
folder = sprintf('TESTDOG');
for i = 1:2500
    baseFileName = sprintf('test_dog_ #%d.png', i);
    fullFileName = fullfile(folder, baseFileName);
    imwrite(TEST_DOG_NEW(:,:,:,i), fullFileName);
end