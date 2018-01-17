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
tic;
beta_mc = (X_train' * X_train) \ (X_train' * y_train);
error_mc = norm(beta_vrai - beta_mc)^2;
fprintf('Erreur Moindres Carres: %.2f \n', error_mc);
fprintf('\tTime: %.2f \n', toc);

% on constate un écart assez important entre les beta prédits et les originaux

%% 2. Le Lasso et son dual : vérifiez que les deux méthodes donnent le même résultat.
% a) Résoudre le Lasso comme nous l’avons fait lors du TP1
lambda = 2;
tic;
cvx_begin quiet
  variables beta_primal(p)
  minimize( .5 * (y_train - X_train * beta_primal)' * (y_train - X_train * beta_primal) + lambda * sum(abs(beta_primal)))
cvx_end

% b) Calculez l’erreur du modèle
error_primal = norm(beta_vrai - beta_primal)^2;
fprintf('Erreur Lasso Primal: %.2f \n', error_primal);
fprintf('\tTime: %.2f \n', toc);

% c) résoudre le dual du lasso à l’aide de CVX
tic;
cvx_begin quiet
    variables beta_dual(p)
    dual variable d
    minimize(sum_square(X_train * beta_dual))
    subject to
    d : abs(X_train' * (X_train * beta_dual - y_train)) <= lambda
    0 <= beta_dual
cvx_end
error_dual = norm(beta_vrai - beta_dual)^2;
fprintf('Erreur Lasso Dual: %.2f \n', error_dual);
fprintf('\tTime: %.2f \n', toc);

%% 3. Le lasso adaptatif
% a) Calculez les poids
tic;
w = 1 ./ abs(beta_mc); % set the weight

% b) Résoudre le problème du lasso adaptatif à l’aide de CVX
cvx_begin quiet
    variables beta_adapt_lasso(p)
    minimize(.5 * (y_train - X_train * beta_adapt_lasso)' * (y_train - X_train * beta_adapt_lasso) + lambda * w' * abs(beta_adapt_lasso))
cvx_end

error_adapt_lasso = norm(beta_vrai - beta_adapt_lasso)^2;
fprintf('Erreur Adaptive Primal: %.2f \n', error_adapt_lasso)
fprintf('\tTime: %.2f \n', toc);

% c) Résolution du problème en décomposant B en B+ et B-
tic;
b = w' * beta_adapt_lasso;
cvx_begin quiet
    variables beta_p_cvx(p) beta_m_cvx(p)
    dual variable d
    minimize(sum_square(y_train - X_train * (beta_p_cvx - beta_m_cvx)))
    subject to
       d : w' * (beta_p_cvx + beta_m_cvx) <= b;
       beta_p_cvx >= 0;
       beta_m_cvx >= 0;
cvx_end
beta_cvx = beta_p_cvx - beta_m_cvx;
error_decomposed_adapt_lasso = norm(beta_vrai - beta_cvx)^2;

fprintf('Erreur Adaptive Dual Decompose: %.2f \n', error_decomposed_adapt_lasso)
fprintf('\tTime: %.2f \n', toc);

% d) Résoudre le problème en utilisant votre fonction du TP1 β ← lasso(X, y)
tic;
[~, beta_lasso] = lasso(X_train * diag(1./(w)), y_train);
error_lasso = norm(beta_vrai - beta_lasso)^2;
fprintf('Error Lasso: %.2f \n', error_lasso);
fprintf('\tTime: %.2f \n', toc);

% e) Résoudre le problème du lasso adaptatif dans le dual
tic;
cvx_begin quiet
    cvx_precision best
    variables beta_dual2(p)
    dual variable d
    minimize(sum_square(X_train * beta_dual2))
    subject to
      d : abs(X_train' * (X_train * beta_dual2 - y_train)) <= lambda * w
      beta_dual2 >= 0
cvx_end

error_adapt_lasso2 = norm(beta_vrai - beta_dual2)^2;
fprintf('Erreur Adaptive Lasso Dual: %.2f \n', error_adapt_lasso2);
fprintf('\tTime: %.2f \n', toc);

% g) Résultats
% Erreur Moindres Carres: 27.09
% 	Time: 0.00
% Erreur Lasso Primal: 60.68
% 	Time: 0.40
% Erreur Lasso Dual: 60.67
% 	Time: 0.37
% Erreur Adaptive Primal: 43.83
% 	Time: 0.26
% Erreur Adaptive Dual Decompose: 43.83
% 	Time: 0.32
% Error Lasso: 60.65
% 	Time: 8.52
% Erreur Adaptive Lasso Dual: 43.83
% 	Time: 0.31
%
% Nous pouvons constater que les erreurs obtenues sont égales selon les méthodes.
% Le constat est que le lasso adaptatif est plus performant que le lasso simple,
% mais est toujours moins performant que la méthode des moindres carrés.
