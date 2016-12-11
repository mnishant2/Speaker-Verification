function val=verify(ID,thresh,X)
filename=sprintf('neural_param%02d.mat',ID);
load(filename);
val=0;
m = size(X, 1);
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

if h2>thresh
    val=1;
end

end