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
fprintf('Applying backpropagation...\n');

a=randperm(size(trainingdata3,1));
X=double(trainingdata3(a,:));

x_train=X(:,1:13);
y_train=X(:,14);



nn_per_category=100; % number of neurons per category
num_labels=2;
%==========================node and beta================================

[node ,beta0]= node_beta(x_train,y_train,nn_per_category,num_labels);

distance_sq=dist(x_train,node').^2;%distance b/w each input and each node
                       %operation at layer 1
        m=size(x_train,1);               

activation=exp(-distance_sq.*repmat(beta0,1,m)'); % activation achieved at layer 2 

% activation=[ones(m,1) activation];
% 
% 
% 
% %=========================================================================
% initial_Theta= randInitializeWeights(nn_per_category*num_labels, num_labels);
% 
% 
% 
% % Unroll parameters
% initial_nn_params = initial_Theta(:);
% 
% 
% %==================backpropagation==============================
% options = optimset('GradObj','on','MaxIter',100);
% 
% %  You should also try different values of lambda
% lambda=10;
% 
% % Create "short hand" for the cost function to be minimized
% costFunction = @(p)RBF_ver_costfunc(p,x_train,y_train,activation,nn_per_category,num_labels,lambda);
% 
% % Now, costFunction is a function that takes in only one argument (the
% % neural network parameters)  
% [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
% 
% Theta=reshape(nn_params,nn_per_category*num_labels+1,num_labels);

Theta=(pinv(activation'*activation)*(activation'*y_train));
filename=sprintf('neural_param%02d.mat',i);
save(filename,'Theta','node','beta0');
end
fprintf('end of training\n');