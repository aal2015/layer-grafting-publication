# https://github.com/jayroxis/CKA-similarity/blob/main/CKA.py

# inspired by
# https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.py

import math
import torch
import numpy as np

class CKA(object):
    def __init__(self):
        pass 
    
    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H) 

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX
 
    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

    
class CudaCKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)

from tqdm import tqdm

class CKAEvaluator:
    def __init__(self, device):
        self.cuda_cka = CudaCKA(device)

    def similarity_stats(self, cls_reps_similarity):
        adj_similarities = []
        for i in range(1, len(cls_reps_similarity)):
            adj_similarities.append(cls_reps_similarity[i-1][i])
    
        argmax = int(np.argmax(adj_similarities))
        argmin = int(np.argmin(adj_similarities))
        
        return {
            'average': float(np.mean(adj_similarities)),
            'max': float(adj_similarities[argmax]),
            'argmax': argmax,
            'min': float(adj_similarities[argmin]),
            'argmin': argmin,
            'adj_similarities': adj_similarities
        }

    def _extract_cls_token_embedding(self, hidden_states):
        only_cls_embeddings = []
        for hs in hidden_states:
            cls_embedding = hs[:,0,:]
            only_cls_embeddings.append(cls_embedding)
            
        return only_cls_embeddings
    
    def _pairwise_helper(self, sublayer_outs, only_cls_token=False, is_att = False):
        if is_att:
            sublayer_outs = flatten_atts(sublayer_outs)
        elif only_cls_token:
            sublayer_outs = self._extract_cls_token_embedding(sublayer_outs)
            
        # shape [batch_size, dim1 * .... * dim_n]
        flattened_out_list = []
        for sublayer_out in sublayer_outs:
            flattened_out = torch.flatten(sublayer_out, start_dim=1)
            flattened_out_list.append(flattened_out)
        
        layer_similarity  = []
        layers_similarity = []
        
        for out_1 in flattened_out_list:
            for out_2 in flattened_out_list:
                cka_value = self.cuda_cka.linear_CKA(out_1, out_2)
                layer_similarity.append(cka_value.detach().cpu())
    
            layers_similarity.append(np.array(layer_similarity))
            layer_similarity = []
        
        return np.array(layers_similarity)

    def pairwise(self, model, dataloader, device, only_cls_token=False, max_iter=float("Inf")):
        model.set_use_module_grafting(False)
        model.set_use_scc_status(False)
        
        n_batch = len(dataloader)
        reps_similarity, atts_similarity = None, None
    
        progress_bar = tqdm(range(min(n_batch, max_iter)),desc="CKA Evaluation")
        
        model.eval()
        for step, batch in enumerate(dataloader):
            if step == max_iter:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # get model output
            with torch.no_grad():
                output = model(**batch)
            loss = output.loss
            reps = output.hidden_states['hidden_states'][1:] # omitting embedding output
            # atts = output.attentions
            
            # similarity estimation
            if reps_similarity is None:
                reps_similarity = self._pairwise_helper(reps, only_cls_token)
                # atts_similarity = CKA_pairwise_helper(atts, only_cls_token, is_att=True)
            else:
                reps_similarity += self._pairwise_helper(reps, only_cls_token)
                # atts_similarity =+ CKA_pairwise_helper(atts, only_cls_token, is_att=True)
            
            progress_bar.update(1)
                    
        # averaging
        reps_similarity = reps_similarity / n_batch
        # atts_similarity = atts_similarity / n_batch
        
        # return reps_similarity, atts_similarity
        return reps_similarity