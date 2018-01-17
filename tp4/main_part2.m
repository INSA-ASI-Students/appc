% TP 04 - APPC
% Florian Martin
% Thibault Théologien

addpath('../utils');
clean_env();

load('housing.mat');

%% 2. Sparse Support Vector Machine in the primal.
% a) Iterative proximal gradient algorithm

% choose the penalty value. the larger lambda, the sparser the solution lambda=1;
% compute step size $L$ the norm of the Hessian
augmented_matrix = [X ones(size(X,1), 1)];
step_size = 1 / norm(augmented_matrix' * augmented_matrix);
w = zeros(size(X, 2), 1);
w0 = 0;
lambda = 2;
epsi = 0.01;

% proximal descent algorithm
Y = diag(y);
tic;
for i = 1:5000
  % computing the gradient wrt w and w0
  loss = max(1 - Y * (X * w + w0), 0);
  grad_w = -(Y * X)' * loss;
  grad_w0 = -(Y * ones(size(X, 1), 1))' * loss;
  % proximal step
  w = prox_l1(w - step_size * grad_w, lambda * step_size);
  w0 = w0 - step_size * grad_w0;
end
fprintf('Time: %.2f \n', toc);
% check how good your algorithm is doing on the test set
perf_proximal = mean(sign(xtest * w + w0) == ytest);
fprintf('Performances : %.2f\n', perf_proximal);

% b) Proximal Sparse SVM
tic;
lambda = .5;
epsi = .01;
[w, w0] = proximal_sparse_svm(X, y, lambda, epsi);
fprintf('Time: %.2f \n', toc);

perf_proximal_svm = mean(sign(xtest * w + w0) == ytest);
fprintf('Performances : %.2f\n', perf_proximal_svm);
% On n'atteint pas les conditions d'optimalité donc une erreur doit être présente
