%% Machine Learning Mini-MATLAB Project 2: Bayesian Linear Regression
%% Dennis Gavrilov, Andrew Koe, Miraj Patel
clear all; 
close all; 
clc;

%PART 2:

%generating axis, just like part 1
x = 0:.001:1;
y = 0:.001:1;
stddev = 0.2;
sampleDataT = zeros(size(x));

%real sine function
realSine = sin(2*pi.*x);

%figure; TEST
%plot(x,realSine, 'g');

%forloop to generate sample data
for i=1:1:length(x)
    %statements
    xn = unifrnd(-1,1);
    noise = normrnd(0,stddev);
    sampleDataT(i) = sin(2*pi.*x(:,i)) + (0.1)*noise;
    %keyboard;
end
totalData= [x', sampleDataT'];

%The Hyperparameters:
alpha = 2.0;
beta = (1/stddev)^2;

%Update Eqns
%First for ONE Target:
%Mn update eqn: Mn = beta*Sn*transpose(design)*targets
%Sn update eqn: inverse(Sn) = alpha*Identity + beta*transpose(design)*design

%This time we have 9 basis functions
%textbook said to refer to basis function (3.4)
%phij(x) = exp(-((x-Mu_j)^2)/(2*(s)^2)) where Mu_j governs locations of
%basis functions in input space, and s is the spacial scale

%to make even input space
Mu_j = linspace(0,1,9); %9 Mu's
sValue = 0.2; %arbitrary value for spacial scale
s = 0.2 * (ones(1,9));

%get one sample point
target1 = datasample(totalData,1);
pulledX = target1(:,1); %only want target inputs

phi_j = [];
%forloop to create each basis
for j=1:9
    phi_j = [phi_j; exp(-((pulledX-Mu_j(j)).^2)/(2*(s(j)).^2))];
end

%now want to create design matrix
design = transpose(phi_j);

%update
inverseSn = alpha*eye(9,9) + beta*(transpose(design))*design; %with 9 basis functions, eye is always 9x9
Sn = inv(inverseSn);
Mn = beta*Sn*(transpose(design))*target1(:,2); %only want target outputs

%Create phi function that can be used to calculate variance
for ll=1:9
    phi_x(ll,:) = exp(-((x-Mu_j(ll)).^2)/(2.*(s(ll)).^2));
end

%Get variance at every point in x's linspace
%Eqn Var = 1/beta + transpose(phi(x))*Sn*phi(x)
for ww=1:length(x)
    variance(ww) = 1/beta + transpose(phi_x(:,ww))*Sn*phi_x(:,ww);
end

%Posterior = Normal(t|transpose(Mn)*phi_j,var())
%Not necessary

figure;
hold on;
scatter(target1(:,1),target1(:,2),80,'MarkerEdgeColor',[0 0 1],...
              'LineWidth',1.5);
predictiveDist = transpose(Mn)*phi_x;
plot(x,realSine, 'g');
plot(x,predictiveDist,'r');

%Shaded part
minusvar=predictiveDist-sqrt(variance);
plusvar=predictiveDist+sqrt(variance);
X=[x,fliplr(x)];
Y=[minusvar,fliplr(plusvar)];
fill(X,Y,'r','FaceAlpha',.2,'EdgeColor','none');
hold off;

%% Now to do it all again but with 2 targets
targets = [];
pulledX = [];
phi_j = [];
for d=1:2
    targets =[targets; datasample(totalData,1)];
    pulledX(d,:) = targets(d,1); %only want target inputs
    %forloop to create each basis
    for j=1:9
        phi_j(d,j) = [exp(-((pulledX(d,1)- Mu_j(j)).^2)/(2*(s(j)).^2))];
    end
end

%now want to create design matrix
design = (phi_j);

%update
inverseSn = alpha*eye(9,9) + beta*(transpose(design))*design; %with 9 basis functions, eye is always 9x9
Sn = inv(inverseSn);
Mn = beta*Sn*(transpose(design))*targets(:,2); %only want target


%Create phi function that can be used to calculate variance
for ll=1:9
    phi_x(ll,:) = exp(-((x-Mu_j(ll)).^2)/(2.*(s(ll)).^2));
end
    
%Get variance at every point in x's linspace
%Eqn Var = 1/beta + transpose(phi(x))*Sn*phi(x)
for ww=1:length(x)
    variance(ww) = 1/beta + transpose(phi_x(:,ww))*Sn*phi_x(:,ww);
end

figure;
hold on;
scatter(targets(:,1),targets(:,2),80,'MarkerEdgeColor',[0 0 1],...
              'LineWidth',1.5);
predictiveDist = transpose(Mn)*phi_x;
plot(x,realSine, 'g');
plot(x,predictiveDist,'r');
%Shaded part
minusvar=predictiveDist-sqrt(variance);
plusvar=predictiveDist+sqrt(variance);
X=[x,fliplr(x)];
Y=[minusvar,fliplr(plusvar)];
fill(X,Y,'r','FaceAlpha',.2,'EdgeColor','none');
hold off;

%% Now to do it all again but with 4 targets
targets = [];
pulledX = [];
phi_j = [];
for d=1:4
    targets =[targets; datasample(totalData,1)];
    pulledX(d,:) = targets(d,1); %only want target inputs
    %forloop to create each basis
    for j=1:9
        phi_j(d,j) = [exp(-((pulledX(d,1)- Mu_j(j)).^2)/(2*(s(j)).^2))];
    end
end

%now want to create design matrix
design = (phi_j);

%update
inverseSn = alpha*eye(9,9) + beta*(transpose(design))*design; %with 9 basis functions, eye is always 9x9
Sn = inv(inverseSn);
Mn = beta*Sn*(transpose(design))*targets(:,2); %only want target


%Create phi function that can be used to calculate variance
for ll=1:9
    phi_x(ll,:) = exp(-((x-Mu_j(ll)).^2)/(2.*(s(ll)).^2));
end
    
%Get variance at every point in x's linspace
%Eqn Var = 1/beta + transpose(phi(x))*Sn*phi(x)
for ww=1:length(x)
    variance(ww) = 1/beta + transpose(phi_x(:,ww))*Sn*phi_x(:,ww);
end

figure;
hold on;
scatter(targets(:,1),targets(:,2),80,'MarkerEdgeColor',[0 0 1],...
              'LineWidth',1.5);
predictiveDist = transpose(Mn)*phi_x;
plot(x,realSine, 'g');
plot(x,predictiveDist,'r');
%Shaded part
minusvar=predictiveDist-sqrt(variance);
plusvar=predictiveDist+sqrt(variance);
X=[x,fliplr(x)];
Y=[minusvar,fliplr(plusvar)];
fill(X,Y,'r','FaceAlpha',.2,'EdgeColor','none');
hold off;

%% Now to do it all again but with 10 targets
targets = [];
pulledX = [];
phi_j = [];
for d=1:10
    targets =[targets; datasample(totalData,1)];
    pulledX(d,:) = targets(d,1); %only want target inputs
    %forloop to create each basis
    for j=1:9
        phi_j(d,j) = [exp(-((pulledX(d,1)- Mu_j(j)).^2)/(2*(s(j)).^2))];
    end
end

%now want to create design matrix
design = (phi_j);

%update
inverseSn = alpha*eye(9,9) + beta*(transpose(design))*design; %with 9 basis functions, eye is always 9x9
Sn = inv(inverseSn);
Mn = beta*Sn*(transpose(design))*targets(:,2); %only want target


%Create phi function that can be used to calculate variance
for ll=1:9
    phi_x(ll,:) = exp(-((x-Mu_j(ll)).^2)/(2.*(s(ll)).^2));
end
    
%Get variance at every point in x's linspace
%Eqn Var = 1/beta + transpose(phi(x))*Sn*phi(x)
for ww=1:length(x)
    variance(ww) = 1/beta + transpose(phi_x(:,ww))*Sn*phi_x(:,ww);
end

figure;
hold on;
scatter(targets(:,1),targets(:,2),80,'MarkerEdgeColor',[0 0 1],...
              'LineWidth',1.5);
predictiveDist = transpose(Mn)*phi_x;
plot(x,realSine, 'g');
plot(x,predictiveDist,'r');
%Shaded part
minusvar=predictiveDist-sqrt(variance);
plusvar=predictiveDist+sqrt(variance);
X=[x,fliplr(x)];
Y=[minusvar,fliplr(plusvar)];
fill(X,Y,'r','FaceAlpha',.2,'EdgeColor','none');
hold off;

%% Now to do it all again but with 25 targets
targets = [];
pulledX = [];
phi_j = [];
for d=1:25
    targets =[targets; datasample(totalData,1)];
    pulledX(d,:) = targets(d,1); %only want target inputs
    %forloop to create each basis
    for j=1:9
        phi_j(d,j) = [exp(-((pulledX(d,1)- Mu_j(j)).^2)/(2*(s(j)).^2))];
    end
end

%now want to create design matrix
design = (phi_j);

%update
inverseSn = alpha*eye(9,9) + beta*(transpose(design))*design; %with 9 basis functions, eye is always 9x9
Sn = inv(inverseSn);
Mn = beta*Sn*(transpose(design))*targets(:,2); %only want target


%Create phi function that can be used to calculate variance
for ll=1:9
    phi_x(ll,:) = exp(-((x-Mu_j(ll)).^2)/(2.*(s(ll)).^2));
end
    
%Get variance at every point in x's linspace
%Eqn Var = 1/beta + transpose(phi(x))*Sn*phi(x)
for ww=1:length(x)
    variance(ww) = 1/beta + transpose(phi_x(:,ww))*Sn*phi_x(:,ww);
end

figure;
hold on;
scatter(targets(:,1),targets(:,2),80,'MarkerEdgeColor',[0 0 1],...
              'LineWidth',1.5);
predictiveDist = transpose(Mn)*phi_x;
plot(x,realSine, 'g');
plot(x,predictiveDist,'r');
%Shaded part
minusvar=predictiveDist-sqrt(variance);
plusvar=predictiveDist+sqrt(variance);
X=[x,fliplr(x)];
Y=[minusvar,fliplr(plusvar)];
fill(X,Y,'r','FaceAlpha',.2,'EdgeColor','none');
hold off;

%finally done