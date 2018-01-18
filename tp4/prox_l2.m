function w = proxl2(lambda, alpha, X)
  w = max((sign(X) .* (abs(X) - lambda * alpha)) / (1 + 2 * lambda * (1 - alpha)), 0);
end
