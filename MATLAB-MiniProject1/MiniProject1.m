%% Machine Learning Mini-MATLAB Project 1
%% Dennis Gavrilov, Andrew Koe, Miraj Patel
%% Part 1a) Sequential Maximum Likelihood (ML) Estimate for a Binomial R.V.
% Randomly choose a success probability between 0.2-0.8
BIN_unknownMu = rand(1)*.6 + 0.2;
numberOfObservations = 250;
numberOfIterations = 1000;
BIN_MLE_means = zeros(numberOfObservations, numberOfIterations);
BIN_MSE_mu = zeros(numberOfObservations, numberOfIterations);
BIN_MLE_means = zeros(numberOfObservations, numberOfIterations);
BIN_MSE_mu = zeros(numberOfObservations, 1);

% Generate observations from Bernoulli trials where probability of success
% is mu (success = 1, failure = 0)
BIN_x = rand(numberOfObservations, numberOfIterations) < BIN_unknownMu;
BIN_muML = BIN_x(1, :);
BIN_MLE_means(1, :) = BIN_muML;

% Sequentially update the estimate of the mu based on received X's
for i = 2:numberOfObservations
    % Compute new ML estimate of the mu using the next X(n) sample
    BIN_muML = BIN_muML + (1/i)*(BIN_x(i, :) - BIN_muML);

    % Insert ML mean for the nth sample size
    BIN_MLE_means(i, :) = BIN_muML;
end

% Compute Mean Squared Error of the mu-ML at each sample size
for i = 1:numberOfObservations
    sqerr = (BIN_MLE_means(i, :) - BIN_unknownMu).^2;
    % Average the squared error across all iterations
    BIN_MSE_mu(i) = sum(sqerr)/numberOfIterations;
end

h1 = figure(1);
t_sample = 1:numberOfObservations;
plot(t_sample, BIN_MSE_mu)
title('MSE of Sequential \mu_{ML} of Binomial R.V. X')
legend('MSE of \mu_{ML} using N samples')
xlabel('Number of samples of X')
ylabel('Mean Squared Error (Variance)')

%% Part 1b) Sequential Maximum Likelihood (ML) Estimate for a Gaussian R.V.
% Randomly choose a mean between 2-5, and Variance between 0.5-1.5
GAUS_unknownMu = rand(1)*3 + 2;
GAUS_unknownVariance = rand(1) + 0.5;
GAUS_unknownSigma = sqrt(GAUS_unknownVariance);
numberOfObservations = 250;
numberOfIterations = 1000;
GAUS_MLE_means = zeros(numberOfObservations, numberOfIterations);
GAUS_MSE_mu = zeros(numberOfObservations, 1);

% Generate N = numberofObservations samples of X
GAUS_x = normrnd(GAUS_unknownMu, GAUS_unknownSigma, numberOfObservations, numberOfIterations);
GAUS_muML = GAUS_x(1, :);
GAUS_MLE_means(1, :) = GAUS_muML;

% Sequentially update the estimate of the mu based on received X's
for i = 2:numberOfObservations
    % Compute new ML estimate of the mu using the next X(n) sample
    GAUS_muML = GAUS_muML + (1/i)*(GAUS_x(i, :) - GAUS_muML);

    % Insert ML mean for the nth sample size
    GAUS_MLE_means(i, :) = GAUS_muML;
end

% Compute Mean Squared Error of the mu-ML at each sample size
for i = 1:numberOfObservations
    sqerr = (GAUS_MLE_means(i, :) - GAUS_unknownMu).^2;
    % Average the squared error across all iterations
    GAUS_MSE_mu(i) = sum(sqerr)/numberOfIterations;
end

h2 = figure(2);
t_sample = 1:numberOfObservations;
plot(t_sample, GAUS_MSE_mu)
title('MSE of Sequential \mu_{ML} of Gaussian R.V. X')
legend('MSE of \mu_{ML} using N samples')
xlabel('Number of samples of X')
ylabel('Mean Squared Error (Variance)')

%% Part 2a) Sequential Conjugate Prior Estimate for mu of the Binomial R.V.
span = 0:.01:1;
numberOfObservations = 250;
% Start by letting alpha = 1 and beta = 1 (uniform and equal probability
% success/failure)
BIN_hyperA = 1;
BIN_hyperB = 1;
BIN_hyper = zeros(numberOfObservations/5 + 1, 2);
BIN_hyper(1, :) = [1 1];

% Generate a vector of samples from a Binomial distribution (N Bernoulli
% trials of successes and failures)
BIN_x = rand(numberOfObservations, 1) < BIN_unknownMu;
span = 0:0.001:1;
BIN_priorProbMu = betapdf(span, BIN_hyper(1, 1), BIN_hyper(1, 2));

h3 = figure(3);
plot(span,BIN_priorProbMu);
text(BIN_unknownMu, 0, '\mu', 'VerticalAlignment', 'top', 'FontWeight', 'bold')
legend('\alpha = 1, \beta = 1')

for i=1:numberOfObservations
    if (BIN_x(i) == 1)
        BIN_hyperA = BIN_hyperA + 1;
    else
        BIN_hyperB = BIN_hyperB + 1;
    end
    
    if (mod(i,5) == 0)
        BIN_hyper(i/5 + 1, 1) = BIN_hyperA;
        BIN_hyper(i/5 + 1, 2) = BIN_hyperB;
        BIN_priorProbMu = betapdf(span, BIN_hyper(i/5 + 1, 1), BIN_hyper(i/5 + 1, 2));
        plot(span, BIN_priorProbMu);
        text(BIN_unknownMu, 0, '\mu', 'VerticalAlignment', 'top', 'FontWeight', 'bold')
        legend(['\alpha = ' int2str(BIN_hyper(i/5 + 1, 1)) ', \beta = ' int2str(BIN_hyper(i/5 + 1, 2))]);
        drawnow;
    end
end

%% Part 2b) Sequential Conjugate Prior Estimate for mu of the Gaussian R.V.
clc;
numberOfObservations = 50;
gaussianSamples = sum(GAUS_x, 2)/numberOfIterations;
mu_naught = zeros(numberOfObservations, 1);
lambda = zeros(numberOfObservations, 1);
GAUS_hyperA = zeros(numberOfObservations, 1);
GAUS_hyperB = zeros(numberOfObservations, 1);

sampleMean = sum(gaussianSamples(1:1))/1;
mu_naught(1) = gaussianSamples(1);
lambda(1) = 0;
GAUS_hyperA(1) = 1;
GAUS_hyperB(1) = 1;

% Tau = precision of the X, = 1/(sample variance)
%tau = 1/std(gaussianSamples);
[spanx, spantau] = meshgrid(-8:.1:8);
h4 = figure(4);
for n=2:numberOfObservations
    sampleMean = sum(gaussianSamples(1:n))/n;
    mu_naught(n) = (lambda(1)*mu_naught(1) + n*sampleMean)/(lambda(1)+n);
    lambda(n) = lambda(1) + n;
    GAUS_hyperA(n) = GAUS_hyperA(1) + n/2;
    GAUS_hyperB(n) = GAUS_hyperB(1) + (n/2)*sum((gaussianSamples(1:n)-sampleMean).^2) + (n*lambda(1)*(sampleMean-mu_naught(1)).^2)/(2*(lambda(1) + n));
    
    
    hyperMu = mu_naught(n);
    hyperLambda = lambda(n);
    hyperAlpha = GAUS_hyperA(n);
    hyperBeta = GAUS_hyperB(n);
    
    termone = ((hyperBeta.^hyperAlpha)*sqrt(hyperLambda))/(gamma(hyperAlpha)*sqrt(2*pi));
    termtwo = tau^(hyperAlpha-0.5);
    termthree = exp((-1)*hyperBeta*spantau);
    termfour = exp((-1)*(hyperLambda*tau*(spanx-hyperMu).^2)/2);
    spany = termone*termtwo*termthree*termfour;
    surface(spanx, spantau, spany)
    drawnow;
    view(3)
end

