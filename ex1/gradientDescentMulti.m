function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

for i=1:size(X,2)
    grad(i) = ( sum((X*theta - y).*X(:,i)) / m );
end

for j=1:size(X,2)   
    temp(j) = theta(j) - alpha * (1/m) * sum((X*theta - y).*X(:,j));
end
theta = temp';
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta)

end

end
