import torch
import torch.nn as nn
from torch_geometric.utils import scatter

NODE_FEATURE_DIM = 13
EDGE_RAW_DIM = 1
DEFAULT_Y_SCALE = 25.0


class DirectedEdgeConv(nn.Module):
    """
    Directed edge graph convolution layer based on DISTRICTNET paper eq(2).
    Logic: The update of edge e(u->v) depends on itself and aggregates in-edges of node u 
    and out-edges of node v.
    """
    def __init__(self, hidden_dim):
        super(DirectedEdgeConv, self).__init__()
        # Corresponds to W0 (self-transformation) and W1 (neighbor-transformation) in the paper
        self.W_self = nn.Linear(hidden_dim, hidden_dim)
        # Extension for directed graphs: distinguish aggregation weights for In-edges and Out-edges
        self.W_in = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, edge_attr, edge_index, num_nodes):
        # edge_index: [2, num_edges], first row is src, second row is dst
        src, dst = edge_index

        # 1. Self-transformation of current edge features: W0 * h_e
        h_self = self.W_self(edge_attr)

        # 2. Aggregate predecessor edge features (Incoming Edges)
        # Which edges point to the current source node src?
        # We calculate the average of edge features for each node as a dst,
        # which represents the "In-edge aggregated features" of that node.
        node_in_feats = scatter(edge_attr, dst, dim=0, dim_size=num_nodes, reduce='mean')
        # Get predecessor features for the current source node src
        agg_in = node_in_feats[src]

        # 3. Aggregate successor edge features (Outgoing Edges)
        # Which edges point from the current target node dst?
        # Calculate the average edge features for each node as a src,
        # representing the "Out-edge aggregated features" of that node.
        node_out_feats = scatter(edge_attr, src, dim=0, dim_size=num_nodes, reduce='mean')
        # Get successor features for the current target node dst
        agg_out = node_out_feats[dst]

        # 4. Message passing and update: \sigma(W_0 h_e + W_in \sum h_{in} + W_out \sum h_{out})
        # The summation here provides directed path awareness.
        h_neigh = self.W_in(agg_in) + self.W_out(agg_out)
        out = self.act(h_self + h_neigh)

        return out


class KidneyEdgePredictor(nn.Module):
    """
    Full replica of the DISTRICTNET GNN + DNN architecture.
    """
    def __init__(self, node_feature_dim, edge_raw_dim, hidden_dim=64):
        super(KidneyEdgePredictor, self).__init__()
        
        # ---------------------------------------------------------
        # Stage 1: Initial Edge Embedding
        # Concatenate src node feats, dst node feats, and raw edge feats (e.g., utility)
        # ---------------------------------------------------------
        concat_dim = node_feature_dim * 2 + edge_raw_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # ---------------------------------------------------------
        # Stage 2: GNN Backbone
        # Paper Appendix A.4: "three graph convolution layers, each with a hidden size of 64"
        # ---------------------------------------------------------
        self.conv1 = DirectedEdgeConv(hidden_dim)
        self.conv2 = DirectedEdgeConv(hidden_dim)
        self.conv3 = DirectedEdgeConv(hidden_dim)
        
        # ---------------------------------------------------------
        # Stage 3: MLP Decoder (DNN Prediction Head)
        # Paper Appendix A.4: "two layers 64->64, one layer 64->32, final layer 32->1. All LeakyReLU"
        # ---------------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)  # Final layer with no activation, outputting continuous weight w_hat
        )

    def forward(self, x, edge_index, raw_edge_attr):
        """
        x: [Num_Nodes, node_feature_dim] (Node feature tensor)
        edge_index: [2, Num_Edges] (Edge topology)
        raw_edge_attr: [Num_Edges, edge_raw_dim] (Initial edge attributes, such as utility)
        """
        num_nodes = x.size(0)
        src, dst = edge_index
        
        # 1. Concatenate to build initial edge vector: [E, D_node*2 + D_edge] -> [E, 64]
        edge_attr = torch.cat([x[src], x[dst], raw_edge_attr], dim=-1)
        h_e = self.edge_encoder(edge_attr)
        
        # 2. Three layers of directed edge convolution (dimensionality remains [E, 64])
        h_e = self.conv1(h_e, edge_index, num_nodes)
        h_e = self.conv2(h_e, edge_index, num_nodes)
        h_e = self.conv3(h_e, edge_index, num_nodes)
        
        # 3. Predict edge weights via MLP: [E, 64] -> [E, 1]
        weight_pred = self.mlp(h_e)
        
        return weight_pred.squeeze(-1) # return shape [E]


class MLPBaseline(nn.Module):
    def __init__(self, node_dim=NODE_FEATURE_DIM, edge_dim=EDGE_RAW_DIM, hidden_dim=256):
        super(MLPBaseline, self).__init__()
        input_dim = node_dim * 2 + edge_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, edge_features):
        return self.net(edge_features).squeeze(-1)
