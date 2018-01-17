% TP 02 - APPC
% Florian Martin
% Thibault Théologien
% ASI 5

addpath('../utils');
clean_env();

%% 1. Génération des données du problème.
% a) Générez les données du problème n = 50 et p= 18.
rand('seed', 3);
randn('seed', 3);
n = 50;
p = 18;

X_train = randn(n, p);
C = corrcoef(X_train);
X_train = X_train * C;
X_train = (X_train - ones(n, 1) * mean(X_train)) ./ (ones(n, 1) * std(X_train, 1));
X_train = X_train / sqrt(n);
beta_vrai = zeros(p,1);
beta_vrai(1:10) = [1 2 3 4 5 1 2 3 4 5];
sigma = 0.25;
y_train = X_train * beta_vrai + sigma * randn(n, 1);
[n, p] = size(X_train);

% b) Calculez l’erreur de test de la méthode des moindres carrés
beta_mc = (X_train' * X_train) \ (X_train' * y_train);
error = norm(beta_vrai - beta_mc);

disp(error);
% on constate un écart assez important entre les beta prédits et les originaux

%% 2. Le Lasso et son dual : vérifiez que les deux méthodes donnent le même résultat.
% a) Résoudre le Lasso comme nous l’avons fait lors du TP1
lambda = 2;
cvx_begin
  variables beta1(p)
  minimize( .5 * (y_train - X_train * beta1)' * (y_train - X_train * beta1) + lambda * sum(abs(beta1)))
cvx_end

error1 = norm(beta_vrai - beta1);
fprintf('Lasso Primal: %10.2f - ', error1);
