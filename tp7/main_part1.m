% TP 07 - APPC
% Florian Martin
% Thibault Théologien

addpath('../utils');
clean_env();

%% 1.
n = 200;
p = 2 * n;
T = 5;
rsnr = 30;
epsi = 1e-6;


% e) Ecrire une fonction permettant de tester votre fonction MCP_CW. Vérifiez que le cout
%   diminue et que la solution vérifie bien les conditions d’optimalisés
[X, y, w_opt, indice] = dataset_generator(n, p, T, rsnr);
lambda = .1;
gamma = 2;
beta = [linspace(-10, 10, p/2), zeros(1, p/2)]';
[beta_mcp, cost_evolution] = mcp_cw(X, y, lambda, gamma, beta);

grad = grad_mcp(X, y, beta, lambda, gamma);
grad_mcp = grad_mcp(X, y, beta_mcp, lambda, gamma);

figure();
subplot(2, 1, 1)
plot(grad);
title('Tracé du gradient');

subplot(2, 1, 2)
plot(grad_mcp);
title('Tracé du gradient');

figure();
plot(cost_evolution);
title('Évolution du cout');
% D'après les graphiques on constate que le cout diminue
% La condition d'optimalité (max(abs(beta - b0)) > epsi est atteinte puisque
% l'on sort de la boucle avant le nombre d'itération maximal.
