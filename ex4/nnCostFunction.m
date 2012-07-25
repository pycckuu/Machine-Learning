function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.


% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%X = [ones(m, 1) X];

%X1 = sigmoid(X*Theta1');

%m = size(X1, 1);

%X1 = [ones(m, 1) X1];

%A = sigmoid(X1*Theta2');

%J= ((-y').*log(A)-(1-y').*log(1-A));

%X = [ones(m, 1) X];
%A_1=[ones(size(X,1),1),X];
%Z_2=A_1*Theta1';
%A_2=sigmoid(Z_2);
%A_2=[ones(size(A_2,1),1),A_2];
%Z_3=A_2*Theta2';
%A_3=sigmoid(Z_3);
%yunroll=zeros(m,num_labels);
%for i=1:m
%     yunroll(i,y(i))=1;
%   endfor;
%   computeJ =-(yunroll.*log(A_3) - (1-yunroll).*(1-log(A_3)));
%   J=(1/m)*sum(sum(computeJ,2));
%z2 = X*(Theta1');

%a2 = sigmoid(z2);

%a2 = [ones(size(a2,1),1) a2];
%size(a2);
%z3 = a2*(Theta2');

%a3 = sigmoid(z3);

%Unroll y into a 0 and 1 matrix of 5000x10

%vec_y = eye(num_labels)(y,:);

%Compute J

%J = sum(sum((-vec_y.*log(a3))-((1-vec_y).*log(1-a3))))/m;

a1 = [ones(size(X, 1), 1) X];
a2 = sigmoid(Theta1 * a1');
a2 = [ones(1, size(a2, 2)); a2];
Hx = sigmoid(Theta2 * a2);

yV = repmat(1:num_labels,m,1)==repmat(y,1,num_labels);

aHX = size(Hx) %This is returning 10x5000 like it should
aYV = size(yV) %This is returning 5000x10 like it should


J = (1/m)*sum(sum( log(Hx).*(-yV)' - log(1-Hx).*(1-yV)' ));

% Add ones to the X data matrix
%X = [ones(m, 1) X];
%a2=sigmoid(X*Theta1');
%a2 = [ones(m, 1) a2];
%Hx=sigmoid(a2*Theta2');
%yV = repmat(1:num_labels,m,1)==repmat(y,1,num_labels);
%J = (1/m)*sum(sum( log(Hx).*(-yV)' - log(1-Hx).*(1-yV)' ));


%for iter=1:size(a3,2)	J(:,iter)=(log(a3(:,iter))*(-y)'-log(1-a3(:,iter))*(1-y)')
%	J=(-y)'*log(a3)-(1-y)'*log(1-a3)	
%end
%J=sum(J)/m

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
