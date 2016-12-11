function p =RBF_ver_predict(ID,thres,X)
filename=sprintf('neural_param%02d.mat',ID);
load(filename);
p=0;
distance_sq=dist(X,node').^2;%distance b/w each input and each node
                       %operation at layer 1
        m=size(X,1);               

activation=exp(-distance_sq.*repmat(beta0,1,m)'); % activation achieved at layer 2 
h=activation*Theta;
% activation=[ones(m,1) activation];
% %==================================
% h=sigmoid(activation*Theta);
if h>thres
    p=1;
end
end