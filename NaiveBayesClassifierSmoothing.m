% This Function is written for the course of Machine Learning for Robotics
% at The University of Genova, Assignment 1
% Author: Syed Muhammad Raza Rizvi
% Student ID: 4853521
% 23-May-2021

function [FinalClass, errorate] = NaiveBayesClassifierSmoothing(trainingset, testset, real_class)

 if nargin <2
        disp('Error: Input parameters not sufficient.\n');
        return 
    end
    
    [n, d] = size(trainingset);
    [m, c] = size(testset);
    
    % Checking number of coloumns of the Sets
    if (d ~= c+1) 
        disp('Error: wrong size of the sets.\n');
        return 
    end
    
    % Checking entries of the Training Set
    for i=1:n
        for j=1:d
            if (trainingset(i,j) < 1)
                disp('Error: wrong values of the trainig set.\n');
                return
            end
        end
    end
    
    % Checking entries of the Test Set
    for i=1:m
        for j=1:c
            if (testset(i,j) < 1)
                disp('Error: wrong values of the data set.\n');
                return
            end
        end
    end
    
    %% Training

    [n, d] = size(trainingset);
    [m, c] = size(testset);
    
    a = 1;
    v = max(trainingset);
    
    classes = max(trainingset(:,d));
    variables = d-1;
    maxValX = max(max(trainingset(:,1:end-1)));
    
likelihood = zeros(maxValX, variables, classes);

for i = 1:classes
    checkset = trainingset(trainingset(:,5)==i,:);
    for j = 1:variables
        for k = 1:maxValX
            likelihood(k,j,i) = (sum(checkset(:,j)==k)+ a) /(sum(trainingset(:,d)==i) + a*v(j));
        end
    end
end

P_H = zeros(classes,1);
P_X = zeros(maxValX, variables);

for i = 1:classes
    P_H(i,1) = sum(trainingset(:,d)==i)/n;
end

%% Testing

for i = 1 : m
    for k = 1 : classes
        result = 1;
        for j = 1:c
            if(likelihood(testset(i,j),j,k) > 0)
                result = result * likelihood(testset(i,j),j,k);
            end
        end
        result_M(i,k) = result * P_H(k,1);
    end
end

% Identifying the Class with Max. Probability
maxP = result_M == max(result_M')';
t = (1:classes)';
FinalClass = maxP*t;

%Error Rate
errorate = (sum(FinalClass ~= real_class))/m;

end