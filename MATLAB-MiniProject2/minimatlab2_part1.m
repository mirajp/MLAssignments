%% Machine Learning Mini-MATLAB Project 2: Bayesian Linear Regression
%% Dennis Gavrilov, Andrew Koe, Miraj Patel
clear all; 
close all; 
clc;

%PART 1:
% models the true f(x, a) = a0 + a1*x with additive Gaussian noise
%Generating sample data (20 points) by using sample function:
%f(x,a) = a0 + a1(x) w/ parameter values a0 = -0.3, a1 = 0.5 
% -1<x<1 and adding gauss noise w/ stddev of 0.2

a0 = -0.3;
a1 = 0.5;
x = linspace(-1,1);
y = linspace(-1,1);
stddev = 0.2;
sampleDataT = zeros(size(x));
%forloop to generate sample data
for i=1:1:100
    %statements
    xn = unifrnd(-1,1);
    noise = normrnd(0,stddev);
    sampleDataT(i) = a0 + a1.*xn + noise;
    %keyboard;
end
totalData= [x', sampleDataT'];
%figure; TEST
%plot(x, sampleDataT);

%The Hyperparameters:
alpha = 2.0;
beta = (1/stddev)^2;

%Now to draw 6 samples of the function y(x,w) = w0 + w1*x in which values of w are
%drawn from the prior <- multivariate normal distribution b/c both w0 and
%w1 have normal distributions. Also, since we know nothing, we assume mu=0
%for both univariates, that variance=1, and that correlation is 0 for both univariates (meaning
%covariance matrix is identity)
weight_0 = makedist('Normal'); %set weight0 as normal distribution object
weight_1 = makedist('Normal'); %set weight0 as normal distribution object
mu = [0,0];%means
Sigma = eye(2,2);%covariance matrix
w0 = linspace(-1,1);
w1 = linspace(-1,1);
[W0,W1] = meshgrid(w0,w1);
F = mvnpdf([W0(:) W1(:)],mu,Sigma);
F = reshape(F,length(w1),length(w0));
fig1 = figure(1);
set(fig1, 'Position', [300, 50, 700, 700]);
surf(w0,w1,F,'EdgeColor', 'none', 'FaceColor', 'interp');
%caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
xlabel('w0'); ylabel('w1'); zlabel('Prior/Posterior');
colormap jet;
view(2);

%now to plot the 6 samples
sampleWeight0 = random(weight_0);
sampleWeight1 = random(weight_1);
sampleY = sampleWeight0 + sampleWeight1.*x;
fig2 = figure(2);
set(fig2, 'Position', [300, 50, 700, 700]);
hold on;
plot(x,sampleY, 'LineWidth', 2);
%loop to plot
for c=2:6
    sampleWeight0 = random(weight_0);
    sampleWeight1 = random(weight_1);
    sampleY = sampleWeight0 + sampleWeight1.*x;
    plot(x,sampleY, 'LineWidth', 2);
end
hold off;

%%

%Now to finally make the update equations and update posterior & likelihood
%First for ONE Target:
%Posterior update: p(w|t) = mvn(w|Mn,Sn)
%Mn update eqn: Mn = beta*Sn*transpose(design)*targets
%Sn update eqn: inverse(Sn) = alpha*Identity + beta*transpose(design)*design

%For this part only have two basis functions: phi1 = 1,and phi2 = x
%therefore our y = w0*phi0 + w1*phi1
%Design matrix iota = [phi0(x1), phi1(x1); phi0(x1), phi1(x2);phi0(x1), phi1(x3)...]
phi0 = 1; %phi0(x1) = x^0 -> 1
%phi1 = pulledX
target1 = datasample(totalData,1);
pulledX = target1(:,1); %only want target inputs
phi1 = pulledX;
design = [phi0, phi1];
inverseSn = alpha*eye(2,2) + beta*(transpose(design))*design; %with two basis functions, eye is always 2x2
Sn = inv(inverseSn);
Mn = beta*Sn*(transpose(design))*target1(:,2); %only want target outputs
posterior = mvnpdf([W0(:) W1(:)],transpose(Mn),Sn);
posterior = reshape(posterior,length(w1),length(w0));

%Now the likelihood = p(t|w,
Xn = unifrnd(-1,1);
likelihood = normpdf((target1(:,2)),(W0+W1*Xn),(1/beta));

fig3 = figure(3);
set(fig3, 'Position', [300, 50, 700, 700]);
subplot(3,1,1);
surf(w0,w1,likelihood,'EdgeColor', 'none', 'FaceColor', 'interp');
%xlabel('w0'); ylabel('w1'); zlabel('Prior/Posterior');
colormap jet;
view(2);
title('Last Likelihood');

subplot(3,1,2);
surf(w0,w1,posterior,'EdgeColor', 'none', 'FaceColor', 'interp');
%xlabel('w0'); ylabel('w1'); zlabel('Prior/Posterior');
colormap jet;
view(2);
title('Posterior After One Target Sampled');

%Now pulling 6 fresh new samples
pullWeight = mvnrnd(transpose(Mn),Sn,1);
sampleWeight0 = pullWeight(:,1);
sampleWeight1 = pullWeight(:,2);
sampleY = sampleWeight0 + sampleWeight1.*x;
subplot(3,1,3);
hold on;
plot(x,sampleY, 'r', 'LineWidth', 2);
scatter(target1(:,1),target1(:,2),80,'MarkerEdgeColor',[0 .5 .5],...
              'LineWidth',1.5);
%loop to plot
for c=2:6
    pullWeight = mvnrnd(transpose(Mn),Sn,1);
    sampleWeight0 = pullWeight(:,1);
    sampleWeight1 = pullWeight(:,2);
    sampleY = sampleWeight0 + sampleWeight1.*x;
    plot(x,sampleY, 'r', 'LineWidth', 2);
end
hold off;

%%

%Now to do it all over again, but with 2 targets
%Posterior update: p(w|t) = mvn(w|Mn,Sn)
%Mn update eqn: Mn = beta*Sn*transpose(design)*targets
%Sn update eqn: inverse(Sn) = alpha*Identity + beta*transpose(design)*design

%For this part only have two basis functions: phi1 = 1,and phi2 = x
%therefore our y = w0*phi0 + w1*phi1
%Design matrix iota = [phi0(x1), phi1(x1); phi0(x1), phi1(x2);phi0(x1), phi1(x3)...]
targets = [];
pulledX = [];
phi0 = [];
phi1 = [];
for d=1:2
    targets =[targets; datasample(totalData,1)];
    phi0 = [phi0;1]; %phi0(x1) = x^0 -> 1
    %phi1 = pulledX
    %target1 = datasample(totalData,1);
    pulledX(d,:) = targets(d,1); %only want target inputs
    phi1 = pulledX;
    design = [phi0, phi1];
    inverseSn = alpha*eye(2,2) + beta*(transpose(design))*design; %with two basis functions, eye is always 2x2
    Sn = inv(inverseSn);
    Mn = beta*Sn*(transpose(design))*targets(:,2); %only want target outputs
    posterior = mvnpdf([W0(:) W1(:)],transpose(Mn),Sn);
    posterior = reshape(posterior,length(w1),length(w0));
    
    %Now the likelihood = p(t|w,
    Xn = unifrnd(-1,1);
    likelihood = normpdf((targets(d,2)),(W0+W1*Xn),(1/beta));
end

fig4 = figure(4);
set(fig4, 'Position', [300, 50, 700, 700]);
subplot(3,1,1);
surf(w0,w1,likelihood,'EdgeColor', 'none', 'FaceColor', 'interp');
%xlabel('w0'); ylabel('w1'); zlabel('Prior/Posterior');
colormap jet;
view(2);
title('Last Likelihood');

subplot(3,1,2);
surf(w0,w1,posterior,'EdgeColor', 'none', 'FaceColor', 'interp');
%xlabel('w0'); ylabel('w1'); zlabel('Prior/Posterior');
colormap jet;
view(2);
title('Posterior After 2 Targets Sampled');

%Now pulling 6 fresh new samples
pullWeight = mvnrnd(transpose(Mn),Sn,1);
sampleWeight0 = pullWeight(:,1);
sampleWeight1 = pullWeight(:,2);
sampleY = sampleWeight0 + sampleWeight1.*x;
subplot(3,1,3);
hold on;
plot(x,sampleY, 'r', 'LineWidth', 2);
scatter(targets(:,1),targets(:,2),80,'MarkerEdgeColor',[0 .5 .5],...
              'LineWidth',1.5);
%loop to plot
for c=2:6
    pullWeight = mvnrnd(transpose(Mn),Sn,1);
    sampleWeight0 = pullWeight(:,1);
    sampleWeight1 = pullWeight(:,2);
    sampleY = sampleWeight0 + sampleWeight1.*x;
    plot(x,sampleY, 'r', 'LineWidth', 2);
end
hold off;

%%

%Now to do it all over again, but with 4 targets
%Posterior update: p(w|t) = mvn(w|Mn,Sn)
%Mn update eqn: Mn = beta*Sn*transpose(design)*targets
%Sn update eqn: inverse(Sn) = alpha*Identity + beta*transpose(design)*design

%For this part only have two basis functions: phi1 = 1,and phi2 = x
%therefore our y = w0*phi0 + w1*phi1
%Design matrix iota = [phi0(x1), phi1(x1); phi0(x1), phi1(x2);phi0(x1), phi1(x3)...]
targets = [];
pulledX = [];
phi0 = [];
phi1 = [];
for d=1:4
    targets =[targets; datasample(totalData,1)];
    phi0 = [phi0;1]; %phi0(x1) = x^0 -> 1
    %phi1 = pulledX
    %target1 = datasample(totalData,1);
    pulledX(d,:) = targets(d,1); %only want target inputs
    phi1 = pulledX;
    design = [phi0, phi1];
    inverseSn = alpha*eye(2,2) + beta*(transpose(design))*design; %with two basis functions, eye is always 2x2
    Sn = inv(inverseSn);
    Mn = beta*Sn*(transpose(design))*targets(:,2); %only want target outputs
    posterior = mvnpdf([W0(:) W1(:)],transpose(Mn),Sn);
    posterior = reshape(posterior,length(w1),length(w0));
    
    %Now the likelihood = p(t|w,
    Xn = unifrnd(-1,1);
    likelihood = normpdf((targets(d,2)),(W0+W1*Xn),(1/beta));
end

fig5 = figure(5);
set(fig5, 'Position', [300, 50, 700, 700]);
subplot(3,1,1);
surf(w0,w1,likelihood,'EdgeColor', 'none', 'FaceColor', 'interp');
%xlabel('w0'); ylabel('w1'); zlabel('Prior/Posterior');
colormap jet;
view(2);
title('Last Likelihood');

subplot(3,1,2);
surf(w0,w1,posterior,'EdgeColor', 'none', 'FaceColor', 'interp');
%xlabel('w0'); ylabel('w1'); zlabel('Prior/Posterior');
colormap jet;
view(2);
title('Posterior After 4 Targets Sampled');

%Now pulling 6 fresh new samples
pullWeight = mvnrnd(transpose(Mn),Sn,1);
sampleWeight0 = pullWeight(:,1);
sampleWeight1 = pullWeight(:,2);
sampleY = sampleWeight0 + sampleWeight1.*x;
subplot(3,1,3);
hold on;
plot(x,sampleY, 'r', 'LineWidth', 2);
scatter(targets(:,1),targets(:,2),80,'MarkerEdgeColor',[0 .5 .5],...
              'LineWidth',1.5);
%loop to plot
for c=2:6
    pullWeight = mvnrnd(transpose(Mn),Sn,1);
    sampleWeight0 = pullWeight(:,1);
    sampleWeight1 = pullWeight(:,2);
    sampleY = sampleWeight0 + sampleWeight1.*x;
    plot(x,sampleY, 'r', 'LineWidth', 2);
end
hold off;

%%

%Now to do it all over again, but with 10 targets
%Posterior update: p(w|t) = mvn(w|Mn,Sn)
%Mn update eqn: Mn = beta*Sn*transpose(design)*targets
%Sn update eqn: inverse(Sn) = alpha*Identity + beta*transpose(design)*design

%For this part only have two basis functions: phi1 = 1,and phi2 = x
%therefore our y = w0*phi0 + w1*phi1
%Design matrix iota = [phi0(x1), phi1(x1); phi0(x1), phi1(x2);phi0(x1), phi1(x3)...]
targets = [];
pulledX = [];
phi0 = [];
phi1 = [];
for d=1:10
    targets =[targets; datasample(totalData,1)];
    phi0 = [phi0;1]; %phi0(x1) = x^0 -> 1
    %phi1 = pulledX
    %target1 = datasample(totalData,1);
    pulledX(d,:) = targets(d,1); %only want target inputs
    phi1 = pulledX;
    design = [phi0, phi1];
    inverseSn = alpha*eye(2,2) + beta*(transpose(design))*design; %with two basis functions, eye is always 2x2
    Sn = inv(inverseSn);
    Mn = beta*Sn*(transpose(design))*targets(:,2); %only want target outputs
    posterior = mvnpdf([W0(:) W1(:)],transpose(Mn),Sn);
    posterior = reshape(posterior,length(w1),length(w0));
    
    %Now the likelihood = p(t|w,
    Xn = unifrnd(-1,1);
    likelihood = normpdf((targets(d,2)),(W0+W1*Xn),(1/beta));
end

fig6 = figure(6);
set(fig6, 'Position', [300, 50, 700, 700]);
subplot(3,1,1);
surf(w0,w1,likelihood,'EdgeColor', 'none', 'FaceColor', 'interp');
%xlabel('w0'); ylabel('w1'); zlabel('Prior/Posterior');
colormap jet;
view(2);
title('Last Likelihood');

subplot(3,1,2);
surf(w0,w1,posterior,'EdgeColor', 'none', 'FaceColor', 'interp');
%xlabel('w0'); ylabel('w1'); zlabel('Prior/Posterior');
colormap jet;
view(2);
title('Posterior After 10 Targets Sampled');

%Now pulling 6 fresh new samples
pullWeight = mvnrnd(transpose(Mn),Sn,1);
sampleWeight0 = pullWeight(:,1);
sampleWeight1 = pullWeight(:,2);
sampleY = sampleWeight0 + sampleWeight1.*x;
subplot(3,1,3);
hold on;
plot(x,sampleY, 'r', 'LineWidth', 2);
scatter(targets(:,1),targets(:,2),80,'MarkerEdgeColor',[0 .5 .5],...
              'LineWidth',1.5);
%loop to plot
for c=2:6
    pullWeight = mvnrnd(transpose(Mn),Sn,1);
    sampleWeight0 = pullWeight(:,1);
    sampleWeight1 = pullWeight(:,2);
    sampleY = sampleWeight0 + sampleWeight1.*x;
    plot(x,sampleY, 'r', 'LineWidth', 2);
end
hold off;

%%

%Now to do it all over again, but with 20 targets
%Posterior update: p(w|t) = mvn(w|Mn,Sn)
%Mn update eqn: Mn = beta*Sn*transpose(design)*targets
%Sn update eqn: inverse(Sn) = alpha*Identity + beta*transpose(design)*design

%For this part only have two basis functions: phi1 = 1,and phi2 = x
%therefore our y = w0*phi0 + w1*phi1
%Design matrix iota = [phi0(x1), phi1(x1); phi0(x1), phi1(x2);phi0(x1), phi1(x3)...]
targets = [];
pulledX = [];
phi0 = [];
phi1 = [];
for d=1:20
    targets =[targets; datasample(totalData,1)];
    phi0 = [phi0;1]; %phi0(x1) = x^0 -> 1
    %phi1 = pulledX
    %target1 = datasample(totalData,1);
    pulledX(d,:) = targets(d,1); %only want target inputs
    phi1 = pulledX;
    design = [phi0, phi1];
    inverseSn = alpha*eye(2,2) + beta*(transpose(design))*design; %with two basis functions, eye is always 2x2
    Sn = inv(inverseSn);
    Mn = beta*Sn*(transpose(design))*targets(:,2); %only want target outputs
    posterior = mvnpdf([W0(:) W1(:)],transpose(Mn),Sn);
    posterior = reshape(posterior,length(w1),length(w0));
    
    %Now the likelihood = p(t|w,
    Xn = unifrnd(-1,1);
    likelihood = normpdf((targets(d,2)),(W0+W1*Xn),(1/beta));
end

fig7 = figure(7);
set(fig7, 'Position', [300, 50, 700, 700]);
subplot(3,1,1);
surf(w0,w1,likelihood,'EdgeColor', 'none', 'FaceColor', 'interp');
%xlabel('w0'); ylabel('w1'); zlabel('Prior/Posterior');
colormap jet;
view(2);
title('Last Likelihood');

subplot(3,1,2);
surf(w0,w1,posterior,'EdgeColor', 'none', 'FaceColor', 'interp');
%xlabel('w0'); ylabel('w1'); zlabel('Prior/Posterior');
colormap jet;
view(2);
title('Posterior After 20 Targets Sampled');

%Now pulling 6 fresh new samples
pullWeight = mvnrnd(transpose(Mn),Sn,1);
sampleWeight0 = pullWeight(:,1);
sampleWeight1 = pullWeight(:,2);
sampleY = sampleWeight0 + sampleWeight1.*x;
subplot(3,1,3);
hold on;
plot(x,sampleY, 'r', 'LineWidth', 2);
scatter(targets(:,1),targets(:,2),80,'MarkerEdgeColor',[0 .5 .5],...
              'LineWidth',1.5);
%loop to plot
for c=2:6
    pullWeight = mvnrnd(transpose(Mn),Sn,1);
    sampleWeight0 = pullWeight(:,1);
    sampleWeight1 = pullWeight(:,2);
    sampleY = sampleWeight0 + sampleWeight1.*x;
    plot(x,sampleY, 'r', 'LineWidth', 2);
end
hold off;
