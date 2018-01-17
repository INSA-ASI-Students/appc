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

[X, y, w_opt, indice] = dataset_generator(n, p, T, rsnr);
lambda = .1;
gamma = 2;
beta = [linspace(-10, 10, p/2), zeros(1, p/2)]';
[beta_mcp, cost_evolution] = mcp_cw(X, y, lambda, gamma, beta);

grad = grad_mcp(X, y, beta, lambda, gamma);
% grad = grad_mcp(X, y, beta_mcp, lambda, gamma);

figure();
plot(grad);
title('Tracé du gradient');

figure();
plot(cost_evolution);
title('Évolution du cout');

% -------

% g = grad_mcp(Xt,yt,beta,lambda,gam );
%
% figure(1)
% plot(g,'g')
% title('Tracé du gradient')
%
% X = 0;
% y = 0;
% beta = linspace(-2,2,100)';
% for i=1:length(beta)
%     cout(i) = cout_mcp( X,y,beta(i),lambda,gam );
% end
%
% % Résultat obtenu
%
% figure(2)
% plot(cout,'g')
% title('Tracé du cout')
