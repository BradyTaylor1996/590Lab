import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def quantize_whole_model(net, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    cluster_centers = []
    assert isinstance(net, nn.Module)
    layer_ind = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            pass
            """
            Apply quantization for the PrunedConv layer.
            --------------Your Code---------------------
            """
            
            clust_num = 2**bits
            weights = m.conv.weight.data.cpu().numpy()
            
            max_ind = np.unravel_index(np.argmax(weights), weights.shape)
            min_ind = np.unravel_index(np.argmin(weights), weights.shape)
            max_value = weights[max_ind]
            min_value = weights[min_ind]
            init_cent = np.arange(min_value, max_value, (max_value - min_value)/clust_num).reshape(-1,1)
            
            nonzero_ind = np.nonzero(weights)
            nonzero_val = weights[nonzero_ind]
            
            clust_model = KMeans(n_clusters=2**bits, init=init_cent).fit(nonzero_val.reshape(-1,1))
            
            cluster_centers.append(clust_model.cluster_centers_)
            
            quant_weights = []
            all_labels = clust_model.labels_
            centroids = clust_model.cluster_centers_
            for i in range(len(all_labels)):
                quant_weights.append(centroids[all_labels[i]])
            final_weights = np.concatenate(quant_weights)
            
            final = np.zeros(weights.shape)
            final[nonzero_ind] = final_weights
            
            #print(centroids)
            #print(final)
            
            m.conv.weight.data = torch.from_numpy(final).float().to(device)
            
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
        
        elif isinstance(m, PruneLinear):
            """
            Apply quantization for the PrunedLinear layer.
            --------------Your Code---------------------
            """
            clust_num = 2**bits
            weights = m.linear.weight.data.cpu().numpy()
            
            max_ind = np.unravel_index(np.argmax(weights), weights.shape)
            min_ind = np.unravel_index(np.argmin(weights), weights.shape)
            max_value = weights[max_ind]
            min_value = weights[min_ind]
            init_cent = np.arange(min_value, max_value, (max_value - min_value)/clust_num).reshape(-1,1)
            
            nonzero_ind = np.nonzero(weights)
            nonzero_val = weights[nonzero_ind]
            
            clust_model = KMeans(n_clusters=2**bits, init=init_cent).fit(nonzero_val.reshape(-1,1))
            
            cluster_centers.append(clust_model.cluster_centers_)
            
            quant_weights = []
            all_labels = clust_model.labels_
            centroids = clust_model.cluster_centers_
            for i in range(len(all_labels)):
                quant_weights.append(centroids[all_labels[i]])
            final_weights = np.concatenate(quant_weights)
            
            final = np.zeros(weights.shape)
            final[nonzero_ind] = final_weights
            
            #print(centroids)
            #print(final)
            
            m.linear.weight.data = torch.from_numpy(final).float().to(device)
            
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
    
    print("Saving...")
    torch.save(net.state_dict(), "net_after_quantization.pt")
    return np.array(cluster_centers)
