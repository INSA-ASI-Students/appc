function [beta, cost_evolution] = mcp_dc(X, y, lambda, gamma, beta)
  max_iteration = 250;
  tol = 10^-6;
  b0 = beta + 1;
  i = 0;
  [n, p] = size(X);
  cost_evolution = [];

  while((max(abs(beta - b0)) > tol) && (i < max_iteration))
    indice = randperm(p);
    b0 = beta;
    for j = 1:p
      if(abs(beta(indice(j))) <= (gamma * lambda))
        grad = beta(indice(j))^(2) / (2 * gamma);
        w = 1 - abs(beta(indice(j))) / (gamma * lambda);
      else
        grad = lambda * abs(beta(indice(j))) - (gamma * lambda^2) / 2;
        w = 0;
      end
      beta(indice(j)) = sign(grad) * max(0, min((abs(grad) * w - lambda) / (1 - 1 / gamma), abs(grad) * w));
    end
    cost_evolution = [cost_evolution cout_mcp(X, y, beta, lambda, gamma)];
    i = i + 1;
  end
end
