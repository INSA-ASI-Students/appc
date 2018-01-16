function result = error_calculation(X, y, beta)
  result = (y - X * beta)' * (y - X * beta);
end
