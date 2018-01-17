function [exact_on_zeros, exact_on_non_zeros, ind_non_zero] = optimality_conditions_svm(X, y, w, grad, lambda, epsi)
  ind_zero = find(abs(w) < epsi);
  ind_non_zero = find(abs(w) >= epsi);
  exact_on_zeros = 12 * epsi;
  exact_on_non_zeros = 12 * epsi;
  
  if (length(ind_zero) > 0)
    exact_on_zeros = max(abs(grad(ind_zero)) - lambda);
  end

  if (length(ind_non_zero) > 0)
    exact_on_non_zeros = max(abs(grad(ind_non_zero) + lambda * sign(w(ind_non_zero))));
  end
end
