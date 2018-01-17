function wprox = prox_l1(w, lambda)
  wprox = sign(w) .* max(abs(w) - lambda, 0);
end
