function p =RBFNN_predict(X,beta,node,weights)
distance_sq=dist(X,node').^2;%distance b/w each input and each node
                       %operation at layer 1
        m=size(X,1);               

activation=exp(-distance_sq.*repmat(beta,1,m)'); % activation achieved at layer 2 

p=activation*weights;
p=(vec2ind(p')-1)';
%==================================
% p=round(p);
end