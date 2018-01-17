function [ cout ] = cout_mcp(X, y, beta, lambda, gamma)
  cout = .5 * (X * beta - y)' * (X * beta - y) + sum((abs(beta) > gamma * lambda) * gamma * lambda^2 / 2 + (abs(beta) <= gamma * lambda) .* (lambda * abs(beta) - beta.^2 / (2 * gamma)));
end
