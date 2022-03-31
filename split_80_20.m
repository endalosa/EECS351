clear;
clc;
close all;
load('DATA.mat');

%%TRAIN 10,000 each
for i = 1:3750
    TRAIN_CAT = cat(4, TRAIN_CAT, TEST_CAT(:,:,:,i));
end
for i = 1:3750
    TRAIN_DOG = cat(4, TRAIN_DOG, TEST_DOG(:,:,:,i));
end

%%TEST 2,500 each
TEST_CAT_NEW = zeros(200,200,3,1);
TEST_DOG_NEW = zeros(200,200,3,1);
for i = 3751:6250
    TEST_CAT_NEW = cat(4, TEST_CAT_NEW, TEST_CAT(:,:,:,i));
end
for i = 3751:6250
    TEST_DOG_NEW = cat(4, TEST_DOG_NEW, TEST_DOG(:,:,:,i));
end
TEST_DOG_NEW(:,:,:,1) = [];
TEST_CAT_NEW(:,:,:,1) = [];

%% 
save("80_20_DATA.mat", 'TEST_CAT_NEW', 'TEST_DOG_NEW', 'TRAIN_CAT', 'TRAIN_DOG')