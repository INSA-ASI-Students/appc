function [X, y, w_opt, indice] = dataset_generator(n, p, T, rsnr)
  X = randn(n, p);
  X = X ./ (ones(n, 1) * sqrt(sum(X.^2)));

  ind = randperm(size(X, 2));
  indice = ind(1:T);
  weights = randn(T, 1);
  weights = weights + .1 * sign(weights);
  y = X(:, indice) * weights;
  std_noise = std(y) / rsnr;
  y = y + randn(size(y)) .* (ones(n, 1) * std_noise);
  w_opt = zeros(p, 1);
  w_opt(indice) = weights;
end
