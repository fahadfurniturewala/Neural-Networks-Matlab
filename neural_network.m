function neural_network( train,test,layer,units,round )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
clc;
training=importdata(train);
[m,n]=size(training);
ro=str2num(round);
layers=str2num(layer);
h=str2num(units);
highest=max(max(training(:,1:end-1)));
for i=1:m
    for j=1:n-1
        training(i,j)=training(i,j)/highest;
    end
end

un_classes=unique(training(:,n));
num_classes=length(un_classes);


x=ones(m,n);
x(:,2:n)=training(:,1:n-1);

a = -0.05;
b = 0.05;
weights=zeros(num_classes,n);
for i=1:num_classes
    for j=1:n
        r = (b-a).*rand(1,1) + a;
        weights(i,j)=r(1,1);
        %weights(i,j)=0.05;
    end
end

hiddenv=ones(h+1,1);
hiddenw=zeros(num_classes,h+1);
for i=1:num_classes
    for j=1:h+1
        r = (b-a).*rand(1,1) + a;
        hiddenw(i,j)=r(1,1);
        %hiddenw(i,j)=0.05;
    end
end

for k=1:num_classes
    for r=1:ro
        learning_rate=(0.98).^(r-1);
        for i=1:m  
            t=0;
            if un_classes(k,1)==training(i,n)
                t=1;
            end
            if layers>2
                temp=weights(k,:)*(x(i,:))';        
                sigmoid=1/(1+exp(-temp));
                hiddenv(2:end,1)=sigmoid;
                temp3a=hiddenw(k,:)*hiddenv(:,1);  
                temp3=1/(1+exp(-temp3a));
                hidden_delta=0.00;
                for j=1:h+1                    
                    hidden_delta=hidden_delta+((temp3-t)*temp3*(1-temp3)*hiddenw(k,j)*hiddenv(j,1)*(1-hiddenv(j,1)));                    
                    hiddenw(k,j)=hiddenw(k,j)-(learning_rate*(temp3-t)*temp3*(1-temp3)*hiddenv(j,1));                    
                end                
                for j=1:n
                     weights(k,j)=weights(k,j)-(learning_rate*hidden_delta*x(i,j));
                end                
            else    
                temp=weights(k,:)*(x(i,:))';        
                sigmoid=1/(1+exp(-temp));
                for j=1:n
                    temp2=learning_rate*(sigmoid-t)*sigmoid*(1-sigmoid)*x(i,j);
                    weights(k,j)=weights(k,j)-temp2;
                end
            end
        end
    end

end

testing=importdata(test);
[tm,tn]=size(testing);
high=max(max(testing(:,1:end-1)));
for i=1:tm
    for j=1:tn-1
        testing(i,j)=testing(i,j)/high;
    end
end
testx=ones(tm,tn);
testx(:,2:tn)=testing(:,1:tn-1);

totalacc=0.00;
if layers==2
    for i=1:tm
        test_classes=zeros(num_classes,1);
        for j=1:num_classes
            temp1=weights(j,:)*transpose(testx(i,:));
            test_classes(j,1)=1/(1+exp(-temp1));
        end
        pr=max(test_classes);
        pred=find(test_classes==pr);
        predicted=un_classes(pred,1);
        %{
        for j=1:num_classes
            if test_classes(j,1)==pr
                competing=competing+1;
            end
        end

        c_classes=zeros(competing,1);
        c=0;
        for j=1:num_classes
            if test_classes(j,1)==pr
                c_classes(c,1)=un_classes(j,1);
                c=c+1;
            end
        end
        predicted=c_classes(1,1);
        %}
        actual=testing(i,tn);
        accuracy=0.00;
            if predicted==actual
                accuracy=1.00;
            end
        %{
        if ccompeting>1
            if actual==predicted
                accuracy=1.00;
            end
            if actual~=predicted
                present=0;
                for j=1:competing
                    if c_classes(j,1)==actual
                        present=1;
                    end
                end
                if present==1;
                    accuracy=1/competing;
                end
                if present==0
                    accuracy=0.00;
                end
            end
        end
        %}
        fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n',i-1,predicted,pr,actual,accuracy);
        totalacc=totalacc+accuracy;
    end
    totalacc=totalacc/tm;
    fprintf('classification accuracy=%6.4f\n',totalacc);
end

if layers>2
    for i=1:tm
        hiddenlayer=ones(h+1,1);
        test_class=zeros(num_classes,1);
        for j=1:num_classes
            hiddenA = weights(j,:)*testx(i,:)';
            hiddenlayer(2:end,1)=1/(1+exp(-hiddenA));
            
            a = hiddenw(j,:)*hiddenlayer;
            test_class(j,1)=1/(1+exp(-a));
            
        end
        pr=max(test_class);
        pred=find(test_class==pr);
        predicted=un_classes(pred,1);
        actual=testing(i,tn);
        accuracy=0.00;
        if predicted==actual
            accuracy=1.00;
        end
        fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n',i-1,predicted,pr,actual,accuracy);
        totalacc=totalacc+accuracy;
    end
    totalacc=totalacc/tm;
    fprintf('classification accuracy=%6.4f\n',totalacc);
end
end

