function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    mat = X*theta - y;
    theta = theta - (alpha/m)*X'*mat;
end

end

%--------------------------------------------------------------------------------------

function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
mat = X * theta;
mat = mat - y;
J = sum(sum(mat.^2));
J = J/(2*m);

end

