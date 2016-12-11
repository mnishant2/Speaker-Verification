load('dataspeakerrec5k.mat')
load('labelspeakerrec5k.mat');
res=zeros(50,2);

for j=0:49
    count=0;
    count2=0;
    x=100*j;
for i=1:5000
X=inputdata(i,:);
v=RBF_ver_predict(j,0.6,X);

if v==1
    count2=count2+1;
     if (x+1<=i && i<=x+100)
     count=count+1;
    end
end
end
res(j+1,1)=count;
res(j+1,2)=count2;
end


