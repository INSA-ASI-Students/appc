function [ beta, cost_evolution ] = mcp_cw(X, y, lambda, gamma, beta)
  [n,p] = size(X);
  max_iteration = 250;
  epsi = 10^-6;
  b0 = beta + 1;
  i = 0;

  cost_evolution = [];

  while ((max(abs(beta - b0)) > epsi) && (i < max_iteration))
    indice = randperm(p);
    b0 = beta;
    for j = 1:p
      grad = - (X(:, indice(j))' * (X * beta - y - X(:, indice(j)) * beta(indice(j))));
      beta(indice(j)) = sign(grad) * max(0, min((abs(grad) - lambda) / (1 - 1 / gamma), abs(grad)));
    end
    cost_evolution = [cost_evolution cout_mcp(X, y, beta, lambda, gamma)];
    i = i + 1;
  end
end
