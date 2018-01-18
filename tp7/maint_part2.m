% TP 07 - APPC
% Florian Martin
% Thibault Théologien

addpath('../utils');
clean_env();

n = 30;
p = 50;
X = randn(n,p);
r = .5;

%1/1.25; % model 1 by zhou
for (a = 1:p)
  for b = 1:p
    C(a,b) = r^abs(a-b);
  end
end

X = X * chol(C);
beta = zeros(p, 1);
beta(1:10) = [1 2 3 4 5 -1 -2 -3 -4 -5];
sig = 0.5;
X = (X - ones(n, 1) * mean(X)) ./ (ones(n, 1) * std(X, 1));
X = X / sqrt(n);
y = X * beta + sig * randn(n, 1);
lambda = .5;
gamma = 2;

[beta_mcp_cw, cost_evolution_mcp_cw] = mcp_cw(X, y, lambda, gamma, beta);
[beta_mcp_dc, cost_evolution_mcp_dc] = mcp_dc(X, y, lambda, gamma, beta);

figure();
subplot(3, 1, 1)
plot(grad_mcp(X, y, beta, lambda, gamma));
title('Tracé du gradient');

subplot(3, 1, 2)
plot(grad_mcp(X, y, beta_mcp_cw, lambda, gamma));
title('Tracé du gradient MCP CW');

subplot(3, 1, 3)
plot(grad_mcp(X, y, beta_mcp_dc, lambda, gamma));
title('Tracé du gradient MC DC');

figure();
subplot(2, 1, 1)
plot(cost_evolution_mcp_cw);
title('Évolution du cout MCP CW');

subplot(2, 1, 2)
plot(cost_evolution_mcp_dc);
title('Évolution du cout MCP DC');

% On constate que la méthode mcp_dc atteint plus rapidement ses conditions
% d'optimalité que mcp_cw. De plus le beta obtenu est de meilleure qualité
