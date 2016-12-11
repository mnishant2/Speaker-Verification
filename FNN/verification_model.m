fprintf('Loading Training DATA...\n');

load('dataspeakerrec5k.mat');
load('labelspeakerrec5k.mat');
trainingdata=[inputdata inputlabel];
for i=0:49
    n=100*i;
trainingdata1=trainingdata(n+1:n+100,:);
trainingdata1(:,end)=1;
num_imposter=1000;
a=randperm(size(trainingdata,1));
b=n+1:n+100;
c=setdiff(a,b);
y = datasample(c,num_imposter,'Replace',false);
trainingdata2=trainingdata(y,:);
trainingdata2(:,end)=0;
trainingdata3=[trainingdata1;trainingdata2];
fprintf('Applying backpropagation...');

a=randperm(size(trainingdata3,1));
X=double(trainingdata3(a,:));

x_train=X(:,1:13);
y_train=X(:,14);
%x_test=X(76:100,1:13);
%y_test=X(76:100,14);


%y_train=double(y_train);
input_layer_size=size(x_train,2);
hidden_layer1_size=64;

num_labels=1;

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size);
initial_Theta2 = randInitializeWeights(hidden_layer1_size,num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%==================backpropagation==============================
options = optimset('MaxIter',200);

%  You should also try different values of lambda
lambda = 0.33;

% Create "short hand" for the cost function to be minimized
costFunction = @(p)vericost(p, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   num_labels, x_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)  
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size + 1))):(0 + (hidden_layer1_size * (input_layer_size + 1)))+ num_labels*(hidden_layer1_size+1)), ...
                  num_labels, (hidden_layer1_size + 1));
  filename=sprintf('neural_param%02d.mat',i);
save(filename,'Theta1','Theta2');
end
fprintf('end of training\n');



          
