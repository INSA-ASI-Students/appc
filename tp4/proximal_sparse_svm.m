function [w, w0] = proximal_sparse_svm(X, y, lambda, epsi)
  [~, p] = size(X);
  step_size = 1 / norm(X' * X);
  Y = diag(y);
  w = zeros(p, 1);
  w0 = 0;

  stop_loop = false;
  i = 1;
  max_iterations = 5000;

  while (i < max_iterations && stop_loop == false)
    loss = max(1 - Y * (X * w + w0), 0);
    grad_w = -(Y * X)' * loss;
    grad_w0 = -(Y * ones(size(X, 1), 1))' * loss;

    % proximal step
    w = prox_l1(w - step_size * grad_w, lambda * step_size);
    w0 = w0 - step_size * grad_w0;

    [exact_on_zeros, exact_on_non_zeros, ind_non_zero] = optimality_conditions_svm(X, y, w, grad_w, lambda, epsi);
    if ((exact_on_zeros < 0 || exact_on_zeros < epsi) && exact_on_non_zeros < epsi)
      disp('end')
      stop_loop = true;
    end
    i = i + 1;
  end
end
