function [X, y] = prepare_housing(filename)
  data = load(filename);

  % make X and y matrices
  [n,d] = size(data);
  X = data(:, 1:d-1);
  y = data(:,d);

  % standardize feature values and center target
  mu_y = mean(y);
  y = y - mu_y;
  [X, ~, ~] = standardize_cols(X);
end
