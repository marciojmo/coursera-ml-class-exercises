function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


minError = inf;

for C_test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
  for sigma_test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    % Train SVM for the test params
    model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test)); 
    % Make predictions with the obtained model and the cross validation set.
    predictions = svmPredict(model, Xval);
    % Calculate the total error for the given prediction
    predictionError = mean(double(predictions ~= yval));

    % Keeps the params that gives the smaller error.
    if predictionError < minError
      minError = predictionError;
      C = C_test;
      sigma = sigma_test;
    endif
  endfor
endfor




% =========================================================================

end
