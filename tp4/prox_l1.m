function w = prox_l1(w, lambda)
  w = sign(w) .* max(abs(w) - lambda, 0);
end
