function [g] = grad_mcp(X, y, beta, lambda, gamma)
  indice0 = find(abs(beta) < sqrt(eps));
  indice1 = 1:length(beta);
  indice2 = find(abs(beta) > lambda * gamma);
  indice1(sort([indice0 ; indice2])) = [];

  gls = X' * (X * beta - y);
  g(indice2) = (indice2);
  g(indice1) = gls(indice1) + lambda * sign(beta(indice1)) - beta(indice1) / gamma;
  g(indice0) = abs(gls(indice0)) > lambda;
end
