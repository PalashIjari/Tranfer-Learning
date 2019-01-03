net=load('alexnet');
%analyzeNetwork(net.alexnet)
net=net.alexnet;
inputSize = net.Layers(1).InputSize;
allImages = imageDatastore('training', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%imdsValidation = imageDatastore('validation', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[trainingImages, testImages] = splitEachLabel(allImages, 0.8, 'randomize');
layers = net.Layers;

layersTransfer = net.Layers(1:end-3);


numClasses = 10;

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingImages, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),testImages);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);

[YPred,scores] = classify(netTransfer,augimdsValidation);

YValidation = testImages.Labels;
accuracy = mean(YPred == YValidation)


idx = randperm(numel(testImages.Files),12);
figure
for i = 1:12
    subplot(4,4,i)
    I = readimage(testImages,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end
