function [X_train, y_train, X_test, y_test] = split(X, y, split_rate)
  [n, ~] = size(X);
  n_train = n * split_rate;

  % Génération d'une liste de nombres entre 1 et n ordonnée aléatoirement
  indices = randperm(n);

  % Split du jeu de données
  X_train = X(indices(1:n_train), :);
  y_train = y(indices(1:n_train));

  X_test = X(indices(n_train + 1:end), :);
  y_test = y(indices(n_train + 1:end));
end
