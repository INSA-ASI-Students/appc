function w = proximal_sparse_elastic_net(X, y, lambda, epsi, alpha)
  [~, p] = size(X);
  w = zeros(p, 1);
  stepsize = 1 / norm(X' * X);

  stop_loop = false;
  i = 1;
  max_iterations = 5000;
  while (i < max_iterations && stop_loop == false)
    grad = -X' * (y - X * w);
    w = w - stepsize * grad;
    w = prox_l2(stepsize * lambda, alpha, w);

    [exact_on_zeros, exact_on_non_zeros, ind_non_zero] = optimality_conditions(X, y, w, lambda, epsi);
    if ((exact_on_zeros < 0 || exact_on_zeros < epsi) && exact_on_non_zeros < epsi)
      stop_loop = true;
    end
    i = i + 1;
  end
end
