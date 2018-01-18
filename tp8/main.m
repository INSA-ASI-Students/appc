% TP 08 - APPC
% Florian Martin
% Thibault Théologien

addpath('../utils');
addpath('./toolbox_dimreduc');
clean_env();

%% 1. Génération des données du problème.
% a) Générez les données du problème en utilisant les données MNIST
load('mnist-app.mat');
load('mnist-test.mat');

na = 100;
Xr = [];
yr = [];
for i = 1:10
  ii = find(Ya == i-1);
  ind(i, :) = ii(1:na);
  Xr = [Xr ; Xa(ii(1:na),:)];
  yr = [yr ; Ya(ii(1:na))];
end

%% 2. ACP
% a) Effectuez une analyse en composantes principales des données MNIST. Pensez-vous
% qu’il faille centrer et/ou réduire les données ?
Xac = Xa - ones(length(Xa), 1) * mean(Xa);
[U, S, V] = svd(Xac, 0);

% b) Visualisez les données en deux dimensions en utilisant un code couleur/forme différent pour chaque classe
c = ['*r'; 'og'; 'xm'; '+c'; 'sb'; 'bd'; 'kp'; 'y^'; 'rv'; 'hm'];
for i = 1:10
  plot(U(ind(i, :), 1), U(ind(i, :), 2), c(i, :));
  text(U(ind(i, :), 1), U(ind(i, :), 2), num2str(i - 1));
end

%% 3. MDS
% a) Effectuez une MDS des données MNIST
D = dist(Xr');
Um = cmdscale(D,2);

% b) Visualisez les données en deux dimensions

% 4. Projection de Sammon
% a) Effectuez une MDS avec une distance de Sammon1
Um = mdscale(D, 2, 'criterion', 'sammon');

% b) Visualisez les données en deux dimensions

%% 5. ISOMAP
% a) Effectuez une projection suivant le critère ISOMAP2
Ui = isomap(D, 2);

% b) Visualisez les données en deux dimensions

%% 6. LLE
% a) Effectuez une projection suivant le critère LLE
rng default % for reproducibility
d = 2;
k = 10; % 1 by class ?
Ull = lle(Xr, k, d);
Ul(indp, :) = Ull';

% b) Visualisez les données en deux dimensions

%% 7. Local MDS Feature Learning
% a) Effectuez une projection suivant le critère LLE4
d = 2;
[Ud, total_cost] = MDS_training(D, d, 10, 0, 1);

% b) Visualisez les données en deux dimensions

%% 8. t-SNE - t-Distributed Stochastic Neighbor Embedding (t-SNE)
% a) Effectuez une projection suivant la méthode tSNE
Ul = tsne(Xr, []);
