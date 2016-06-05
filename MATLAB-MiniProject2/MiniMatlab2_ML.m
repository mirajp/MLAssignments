%% Machine Learning Mini-MATLAB Project 2: Bayesian Linear Regression
%% Dennis Gavrilov, Andrew Koe, Miraj Patel
clear all; close all; clc;

% Part 1: Bayesian learning in a linear basis function model and
% sequential update of the posterior distribution -> straight line fitting
% Assume the linear model to be form: y(x, w) = w0 + w1*x
% models the true f(x, a) = a0 + a1*x with additive Gaussian noise
%a0 = rand()*2 - 1;
%a1 = rand()*2 - 1;
noiseVar = 0.5;
beta = (1/noiseVar)^2;
alpha = 2;

% Before any data points are observed, the prior distribution of w0 and w1
% assumes zero-mean gaussian with variance = noise variance
% Generate bivariate/isotropic Gaussian prior/posterior
% -- meshgrid way --
%[w0, w1] = meshgrid(-1:0.1:1, -1:0.1:1);
%prior = normpdf(w0, 0, sqrt(noiseVar)).*normpdf(w1, 0, sqrt(noiseVar));

% -- without meshgrid --
w0 = -1:0.01:1;
w1 = -1:0.01:1;
prior = (normpdf(w0, 0, sqrt(noiseVar))')*normpdf(w1, 0, sqrt(noiseVar));
posterior = prior;
fig1 = figure(1);
surf(w0, w1, prior, 'EdgeColor', 'none', 'FaceColor', 'interp');
colormap jet;
view(2);
set(fig1, 'Position', [100, 150, 550, 500]);
title('Prior/Posterior Distribution of \it{w}\rm\bf Space (with no data)');
xlabel('\bf\itw_{0}');
ylabel('\bf\itw_{1}');

% Draw 6 pairs of w0 and w1 from this prior/posterior
x = -1:0.01:1;
w0sample = normrnd(0, sqrt(noiseVar));
w1sample = normrnd(0, sqrt(noiseVar));
y = w0sample + w1sample*x;
for i = 2:6
    w0sample = normrnd(0, sqrt(noiseVar));
    w1sample = normrnd(0, sqrt(noiseVar));
    y = [y; w0sample + w1sample*x];
end

fig2 = figure(2);
set(fig2, 'Position', [225, 100, 550, 500]);
hold on;
for i = 1:6
    plot(x, y(i, :), 'LineWidth', 1.5);
end
title('Data Space Of x To y Using 6 Pairs of \itw');
xlabel('\bf\itx'); xlim([-1 1]);
ylabel('\bf\ity'); ylim([-1 1]);
hold off;

%% Part 2: Using sequential observations to estimate a0 and a1
% Using sequential samples of given x and noisy target, determine the
% likelihood of that target: P(t | x, w, beta) = N(t | y(x, w), beta)
% The updated posterior = likelihood of target * previous prior
clc;
numberSamples = 10;
% The input, x, is uniformly distributed from -1 to 1
xInput = rand(numberSamples, 1)*2 - 1;
a0 = -0.3;
a1 = 0.5;
w0 = -1:0.01:1;
w1 = -1:0.01:1;

yOutput = a0 + a1*xInput;
receivedTargets = zeros(1, numberSamples);
    
for iter = 1:numberSamples
    trialInput = xInput(iter);
    %trialInput = 1;
    receivedTarget = yOutput(iter) + normrnd(0, sqrt(noiseVar));
    %receivedTarget = 0;
    receivedTargets(iter) = receivedTarget;
    
    likelihoodPDF = zeros(length(w0), length(w1));
    for i = 1:length(w0)
        for j = 1:length(w1)
            likelihoodPDF(i, j) = normpdf(receivedTarget, w0(i) + w1(j)*trialInput, sqrt(noiseVar));
        end
    end
    
    fig3 = figure(3);
    surf(w0, w1, likelihoodPDF, 'EdgeColor', 'none', 'FaceColor', 'interp');
    colormap jet;
    view(2);
    set(fig3, 'Position', [350, 150, 550, 500]);
    
    posterior = posterior.*likelihoodPDF;
    fig4 = figure(4);
    surf(w0, w1, posterior, 'EdgeColor', 'none', 'FaceColor', 'interp');
    colormap jet;
    view(2);
    set(fig4, 'Position', [475, 100, 550, 500]);
    drawnow
end

