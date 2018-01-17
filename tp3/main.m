% TP 03 - APPC
% Florian Martin
% Thibault Théologien
% ASI 5

addpath('../utils');
clean_env();

% Exercice 1 - Sparsity in linear regression
%% 1. Building the signal to approximate
% a) Build the covariate matrix X : Generate a set of n = 200 data points in dimension
% 2n, randomly distributed on a Gaussian hypersphere
n = 200;
p = 2 * n;
X = randn(n, p);
X = X ./ (ones(n, 1) * sqrt(sum(X.^2)));
epsi = 1e-6;

% b) Création d'un signal
T = 5;
rsnr = 30;
ind = randperm(size(X, 2));
indice = ind(1:T);
weights = randn(T, 1);
weights = weights + .1 * sign(weights);
y = X(:, indice) * weights;
std_noise = std(y) / rsnr;
y = y + randn(size(y)) .* (ones(n, 1) * std_noise);
w_opt = zeros(p, 1);
w_opt(indice) = weights;

%% 2. Solving the optimization problem
% a) Implement in CVX the optimization problem above

% Create and solve problem in CVX
lambda = .1;

tic;
cvx_begin
  variable w_cvx(p)
  minimize(.5 * (X * w_cvx - y)' * (X * w_cvx - y) + lambda * sum(abs(w_cvx)))
cvx_end
fprintf('Time: %.2f \n', toc);

% b) Check if the problem has been properly solved by verifying the optimality conditions)
[exact_on_zeros, exact_on_non_zeros, ind_non_zero] = optimality_conditions(X, y, w_cvx, lambda, epsi);
fprintf('Exact on zeros : %.2f \n', exact_on_zeros);
fprintf('Exact on non zeros : %.2f \n', exact_on_non_zeros);

% exact_on_zeros < 0
% exact_on_non_zeros < epsi
% Les résultats sont donc cohérent par rapport à ce qui est attendu

% c) compare the amplitudes of retrieved weights for the true non-zeros elements
% figure();
% plot(w_cvx - w_opt);

% Nous pouvons voir que le lasso n'est pas adapté dans ce cas du fait qu'il ne
% prédit pas correctement les variables attendue. (Forte erreur en des points précis)

% d) play with λ and compare w with w⋆ in terms of support and difference of amplitudes.
% The comparison should hold for the same matrix X and signal y.

nb_non_zeros = length(ind_non_zero); % compute the difference
maxi = max(abs(w_cvx(indice) - w_opt(indice)));
fprintf('lambda: %.2f - T: %d - nbnonzeros: %d - diff: %.3f \n', lambda, T, nb_non_zeros, maxi);

% Si lamda est très petit, l'erreur moyenne est plus faible, mais on se trompe plus souvent.
% Dans le cas contraire, on selectionne moins de variables (on se trompe moins souvent),
% mais l'erreur commise est beaucoup plus importante.
% lambda: 0.50 - T: 5 - nbnonzeros: 2 - diff: 0.558
% lambda: 0.10 - T: 5 - nbnonzeros: 5 - diff: 0.124
% lambda: 0.02 - T: 5 - nbnonzeros: 10 - diff: 0.029
% lambda: 0.01 - T: 5 - nbnonzeros: 21 - diff: 0.014

%% 3. Implementing a coordinatewise approach
% a) for the same problem as above, implement a coordinatewise approach that goes
% through the full data 1000 times

tic;
lambda = .1;
w = zeros(p, 1);
for i = 1:1000
  for k = 1:p
    xk = X(:, k); % Sélection de la variable d'indice k
    w(k) = 0;
    s = y - X * w;
    w(k) = sign(xk' * s) * max(0, abs(xk' * s) - lambda) / (xk' * xk);
  end
end
fprintf('Time: %.2f \n', toc);

% b) Check if the solution is correct through the optimality conditions
[exact_on_zeros, exact_on_non_zeros, ind_non_zero] = optimality_conditions(X, y, w, lambda, epsi);
fprintf('Exact on zeros : %.2f \n', exact_on_zeros);
fprintf('Exact on non zeros : %.2f \n', exact_on_non_zeros);

% c)
tic
w = cd_sparse_regression(X, y, lambda, epsi);
fprintf('Time: %.2f \n', toc);
