function [error, beta] = lasso(X, y)
  [X_train, y_train, X_test, y_test] = split(X, y, .5);

  [~, p] = size(X_train);

  H = [
        X_train' * X_train, -X_train' * X_train;
        -X_train' * X_train, X_train' * X_train
      ];
  c = [
        X_train' * y_train ;
        -X_train' * y_train
      ];
  A = ones(2 * p, 1);
  range = [0 : .5 : 20];
  errors = zeros(size(range));
  beta_list = [];
  for i = 1 : length(range)
    k = range(i);
    cvx_begin quiet
      variables x(2 * p)
      dual variable lambda
      minimize( .5 * x' * H * x - c' * x )
      subject to
        lambda : A' * x <= k
        0 <= x
    cvx_end
    beta = x(1:p) - x(p+1:end);
    errors(i) = error_calculation(X_test, y_test, beta);
    beta_list = [beta_list beta];
  end

  [~, ind] = min(errors);
  error = errors(ind);
  beta = beta_list(:, ind);
end
