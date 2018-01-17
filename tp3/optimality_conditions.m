function [exact_on_zeros, exact_on_non_zeros, ind_non_zero] = optimality_conditions(X, y, w, lambda, epsi)
  ind_zero = find(abs(w) < epsi);
  ind_non_zero = find(abs(w) >= epsi);
  grad = -X' * (y - X * w);
  exact_on_zeros= max(abs(grad(ind_zero)) - lambda);
  exact_on_non_zeros = max(abs(grad(ind_non_zero) + lambda * sign(w(ind_non_zero))));
end
