% TP 04 - APPC
% Florian Martin
% Thibault Théologien

addpath('../utils');
clean_env();

% Exercice 1 - Some proximal operators
%% 1. Proximal operators
% a) Implement a function that implement the l1 operator
% b) Implement a function that computes the group-lasso proximal operator

% Exercice 2 - Proximal gradient algorithms
%% 1. Proximal gradient algorithm for the lasso
% a) Proximal gradient algorithm for the Lasso

% generate dataset
n = 200;
p = 2 * n;
T = 5;
rsnr = 30;
epsi = 1e-6;
[X, y, w_opt, indice] = dataset_generator(n, p, T, rsnr);

% Proximal gradient algorithm
lambda = .1; % choisi arbitrairement
w = zeros(p, 1); % on déclare un vecteur pour les valeurs W
stepsize = 1 / norm(X' * X); % choose L as a step, L being the norm of the Hessian

tic;
for i = 1:5000 %on boucle afin de trouver le gradient et le Woptimale
    grad = -X' * (y - X * w);
    w = w - stepsize * grad;
    w = prox_l1(w, stepsize * lambda);
end

% b) Optimality conditions
[exact_on_zeros, exact_on_non_zeros, ind_non_zero] = optimality_conditions(X, y, w, lambda, epsi);
fprintf('Exact on zeros : %.4f \n', exact_on_zeros);
fprintf('Exact on non zeros : %.4f \n', exact_on_non_zeros);
fprintf('Time: %.2f \n', toc);

% On constate que exact_on_zeros < 0 et que exact_on_non_zeros < epsi,
% les conditions d'optimalité sont donc vérifiées.

% c) Proximal Sparse Regression function
tic;
w = proximal_sparse_regression(X, y, lambda, epsi);
fprintf('Time: %.2f \n', toc);
tic;
w = cd_sparse_regression(X, y, lambda, epsi);
fprintf('Time: %.2f \n', toc);

% d) which algorithm is faster on your problem the coordinate descent algorithm or the
% proximal descent ones? what if we change λ and T .
% Les performances sont relativement équivalentes entre les deux fonctions (cela se joue
% à 0.02s en faveur du cd_sparse_regression en général)
%
% Si l'on diminue lambda, le proximal devient plus rapide.
% À l'inverse le cd_sparse_regression devient plus rapide si lambda est grand.
%
% Concernant T, plus il est grand, plus le proximal est rapide à trouver une solution.
