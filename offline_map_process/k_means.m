% create map cluster centers
clc;clear;close all;

% dataset and sequence
dataset = 'KITTI';
seq = '06';

% parameter
num_clusters = 50;
len = 4; row = 20; col = 60; dist = 3;

% directory
dir_map = strcat('../data/', dataset, '/', seq, '/map/');
map_key_name = strcat(num2str(len), '_', num2str(row), '_', num2str(col), '_', num2str(dist), '.txt');
map_keys = load(strcat(dir_map, 'map_ring_keys_', map_key_name));

% k-means
opts = statset('Display','final');
[ids, centers] = kmeans(map_keys, num_clusters, 'Replicates', 50, 'Options', opts);

% save label and cluster centers
label_name = strcat(dir_map, 'labels_', map_key_name);
cnt_name = strcat(dir_map, 'cnts_', map_key_name);
save(label_name, 'ids', '-ascii');
save(cnt_name, 'centers', '-ascii');


