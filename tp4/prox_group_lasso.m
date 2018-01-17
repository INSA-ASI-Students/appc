function wprox = proxgrouplasso(w, lambda, group)
  % group is a vector of same size of w stating to which group
  % w_i belongs
  nbgroup = max(group); % count the number of group wprox=w(size(w));
  for i = 1:nbgroup
    ind = find(group == i); % find all the variables in a given group
    wprox(ind) = max(1 - lambda / norm(w(ind)), 0) * w(ind);
  end
end
