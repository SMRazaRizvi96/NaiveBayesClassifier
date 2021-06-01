% This code is written for the course of Machine Learning for Robotics
% at The University of Genova, Assignment 1
% Author: Syed Muhammad Raza Rizvi
% Student ID: 4853521
% 23-May-2021

clear;
clc;

%% PREPARING DATASETS

load('Weather_dataset.mat');
Weatherdataset = table2array(Weatherdataset);
[r, c] = size(Weatherdataset);

index = randperm(r);    % Random indexes
m = ceil(r*0.7);        % Taking 70% of the data as Training Set
trainingset = Weatherdataset(index(1:m), :);
testset = Weatherdataset(index(m+1:end), 1:(c-1));
real_class = Weatherdataset(index(m+1:end), c);

%% NAIVE BAYES CLASSIFIER

[FinalClass, errorate] = NaiveBayesClassifier(trainingset, testset, real_class);

%  results
fprintf('error rate without Smoothing: %f\n', errorate);

%% NAIVE BAYES CLASSIFIER WITH SMOOTHING

[FinalClass_s, errorate_s] = NaiveBayesClassifierSmoothing(trainingset, testset, real_class);

%  results
fprintf('error rate with Smoothing: %f\n', errorate_s);