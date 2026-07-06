# Country Similarity Network (real data)

- Nodes: 183 countries; Edges: 524 (k-NN, k=4).
- Communities (greedy modularity): 10; modularity = 0.72.
- Sizes: [31, 23, 21, 21, 21, 15, 13, 13, 13, 12]

## Most central 'bridge' countries (betweenness)
- El Salvador (SLV): betweenness 0.120, community 1
- Greece (GRC): betweenness 0.108, community 2
- Myanmar (MMR): betweenness 0.104, community 1
- Canada (CAN): betweenness 0.085, community 2
- Estonia (EST): betweenness 0.082, community 5
- Lebanon (LBN): betweenness 0.077, community 0

## Outputs
- network_nodes.csv, network_edges.csv
- report_latex/figures/fig_v1_country_network.png