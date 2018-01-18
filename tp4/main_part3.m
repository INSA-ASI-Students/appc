% TP 04 - APPC
% Florian Martin
% Thibault Théologien

addpath('../utils');
clean_env();

load('prostate.mat');
lambda = 0.5;
epsi = 1e-6;
alpha = 0.5; %alpha doit etre [0,1]

tic;
w = proximal_sparse_elastic_net(x_train, y_train, lambda, epsi, alpha);
perf = error_calculation(x_test, y_test, w);
fprintf('Performances : %.2f\n', perf);
fprintf('Time: %.2f \n', toc);

tic;
w = proximal_sparse_regression(x_train, y_train, lambda, epsi);
perf = error_calculation(x_test, y_test, w);
fprintf('Performances : %.2f\n', perf);
fprintf('Time: %.2f \n', toc);

%% La méthode avec elastic net semble plus performante, avec un temps de calcul identique
