% TP 01 - APPC
% Florian Martin
% Thibault Théologien
% ASI 5

clear;
clc;
close all;

randn('seed', 10);

%% 1. Génération des données du problème
% a) Génération des données du problème
% Chargement des données depuis le fichier
[X, y] = prepare_housing('housing.data');

[n, p] = size(X);
q = 5;
p = p + q;

% Ajout de "q" variables aléatoires aux données
X = [X randn(n, q)];

% b) Séparation des données en deux jeux de taille égale
[X_train, y_train, X_test, y_test] = split(X, y, 0.5);

% c) Calculez l’erreur de test de la méthode des moindres carrés
beta_mc = X_train \ y_train;
error_mc = error_calculation(X_test, y_test, beta_mc);

%% 2. Résolution du problème du Lasso
% a) Multiplicateur de Lagrange à l'optimum
k = 10;
tic;
cvx_begin quiet
  cvx_precision best
  variables beta1(p) % Définition d'un vecteur de taille p
  dual variable lambda1 % Définition d'une variable duale
  minimize(norm(y_train - X_train * beta1, 2)) % Terme à minimiser
  subject to
  lambda1 : sum(abs(beta1)) <= k; % Contrainte
cvx_end

% b) Résolution de la formulation du Lasso
cvx_begin quiet
  variables beta2(p)
  minimize(norm(y_train - X_train * beta2, 2) + lambda1 * sum(abs(beta2)))
cvx_end
time1 = toc;

% c) Formulation quadratique du Lasso
tic;
D = X_train' * X_train;
ep = -y_train' * X_train;
cvx_begin quiet
  variables beta3(p)
  dual variable lambda2
  minimize( .5 * beta3' * D * beta3 + ep * beta3 )
  subject to
  lambda2 : sum(abs(beta3)) <= k;
cvx_end
time2 = toc;

% d) e) Programme quadratique
tic;
H = [
      X_train' * X_train, -X_train' * X_train;
      -X_train' * X_train, X_train' * X_train
    ];
c = [
      X_train' * y_train;
      -X_train' * y_train
    ];
A = ones(2 * p, 1);

cvx_begin quiet
  variables x(2 * p)
  dual variable lambda3
  minimize( .5 * x' * H * x - c' * x )
  subject to
  lambda3 : A' * x <= k
  0 <= x
cvx_end
beta3 = x(1:p) - x(p+1:end);
time3 = toc;

disp([beta1 beta2 beta3]);
disp([time1 time2 time3]);

% La méthode la plus rapide est la dernière utilisant CVX sûrement car
% cette méthode permet de converger plus rapidement vers une solution.
% CPlex devrait tout de même être plus rapide, mais nous n'avons pas
% implémenté cette méthode.

error_lasso = error_calculation(X_test, y_test, beta3);
% Le cout de la solution du lasso est moindre que celle de la méthode des moindres carrés

pos = find(abs(beta1) > 0.000001);
beta_mc = X_train(:, pos) \ y_train;
error_mc_lasso = error_calculation(X_test(:, pos), y_test, beta_mc);
disp([error_mc error_lasso error_mc_lasso]);

range = [2 : 0.5 : 30];
errors = zeros(size(range));
B = [];
for i = 1 : length(range)
  k = range(i);
  [xnew, lambda, pos] = monqp(H, c, A, k, inf, 10^-12, 0);
  ind = find(pos > p);
  sign = ones(length(pos),1);
  pos(ind) = pos(ind) - p;
  sign(ind) = -1;
  beta_mc = X_train(:, pos) \ y_train;
  errors(i) = error_calculation(X_test(:, pos), y_test, beta_mc);
  beta = zeros(size(beta1));
  beta(pos) = beta_mc;
  B = [B beta];
end

[v, ind] = min(errors);
beta = B(:, ind);
ind = find(abs(beta) < 0.000001);
X_train(:, ind) = [];
X_test(:, ind) = [];
beta_mc = X_train \ y_train;
error = (y_test - X_test * beta_mc)' * (y_test - X_test * beta_mc);

% On obtient un meilleur résultat avec l'utilisation de la méthode du QP
% par rapport aux méthodes des moindres carrés et du lasso. Cela s'explique
% par le fait que l'on obtient de manière plus optimale les variables qui ont
% de l'importance pour l'apprentissage  avec la méthode du QP.

figure;
subplot(2, 1, 1)
plot(range, errors)
subplot(2, 1, 2)
plot(range, B')

[lasso_error, ~] = lasso(X, y);
