function w = cd_sparse_regression(X, y, lambda, epsi)
  [~, p] = size(X);
  w = zeros(p, 1);

  stop_loop = false;
  i = 1;
  max_iterations = 1000;
  while (i < max_iterations && stop_loop == false)
    for k = 1:p
      xk = X(:, k); % Sélection de la variable d'indice k
      w(k) = 0;
      s = y - X * w;
      w(k) = sign(xk' * s) * max(0, abs(xk' * s) - lambda) / (xk' * xk);
    end
    [exact_on_zeros, exact_on_non_zeros, ind_non_zero] = optimality_conditions(X, y, w, lambda, epsi);
    if ((exact_on_zeros < 0 || exact_on_zeros < epsi) && exact_on_non_zeros < epsi)
      stop_loop = true;
    end
    i = i + 1;
  end
end
