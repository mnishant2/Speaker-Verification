%loading data
load('inputlabel5k.mat');
load('inputdata5k.mat');

%concating inputdata and inputlabel
trainingdata=[inputdata inputlabel]; 
%randomising data
a=randperm(size(inputlabel,1));
X=trainingdata(a,:);


%dividing the data into training and test set
x_train=X(1:3750,1:13);
y_train=X(1:3750,14);
x_test=X(3751:5000,1:13);
y_test=X(3751:5000,14);


nn_per_category=10; % number of neurons per category
num_labels=10;
target_matrix=ind2vec((y_train+1)');

%==========================node and beta================================

[node ,beta]= node_beta(x_train,y_train,nn_per_category,num_labels);

distance_sq=dist(x_train,node').^2;%distance b/w each input and each node
                       %operation at layer 1
        m=size(x_train,1);               
% beta=ones(100,1);
act=exp(-distance_sq.*repmat(beta,1,m)'); % activation achieved at layer 2 
weights=(pinv(act'*act)*(act'*target_matrix'));

pred=RBFNN_predict(x_test,beta,node,weights);

z=(round(y_test-pred)==0);

k=mean(z)*100;
fprintf('Training acuracy is:%f percent',k);


% activation=[ones(m,1) activation];