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

    def flatten_atts(self, attention_outputs):
        """
        attention_outputs: list of tensors, each (batch, heads, seq_len, seq_len)
        flatten heads + attention map -> (batch, heads * seq_len * seq_len)
        """
        flattened = []
        for att in attention_outputs:
            # att: (batch, heads, seq_len, seq_len)
            flattened.append(torch.flatten(att, start_dim=1))  # (batch, heads*seq_len*seq_len)
        return flattened

    def _interleaved_pairwise_helper(self, atts_outs, reps_outs, only_cls_token=False):
        """
        Build a 2N x 2N CKA matrix by interleaving att and ffn sublayer outputs.
        Order: [Att_1, FFN_1, Att_2, FFN_2, ..., Att_N, FFN_N]
        """
        # flatten atts: (batch, heads, seq, seq) -> (batch, heads*seq*seq)
        atts_outs = self.flatten_atts(atts_outs)
    
        # flatten reps: either cls token or full sequence
        if only_cls_token:
            reps_outs = self._extract_cls_token_embedding(reps_outs)
    
        # flatten both to (batch, D)
        atts_flat = [torch.flatten(x, start_dim=1) for x in atts_outs]
        reps_flat = [torch.flatten(x, start_dim=1) for x in reps_outs]
    
        # interleave: [att_0, ffn_0, att_1, ffn_1, ...]
        interleaved = []
        for att, ffn in zip(atts_flat, reps_flat):
            interleaved.append(att)
            interleaved.append(ffn)
    
        # compute full 2N x 2N CKA matrix
        S = len(interleaved)
        layers_similarity = []
        for out_1 in interleaved:
            row = []
            for out_2 in interleaved:
                cka_value = self.cuda_cka.linear_CKA(out_1, out_2)
                row.append(cka_value.detach().cpu())
            layers_similarity.append(np.array(row))
    
        return np.array(layers_similarity)  # (2N, 2N)
    
    def _pairwise_helper(self, sublayer_outs, only_cls_token=False, is_att = False):
        if is_att:
            sublayer_outs = self.flatten_atts(sublayer_outs)
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
        num_steps = min(n_batch, max_iter)
        reps_similarity = reps_similarity / num_steps
        # atts_similarity = atts_similarity / n_batch
        
        # return reps_similarity, atts_similarity
        return reps_similarity

    def subLayer_interleaved_pairwise(self, model, dataloader, device, only_cls_token=False, max_iter=float("Inf")):
        """
        Computes a 2N x 2N interleaved CKA similarity matrix.
        Rows/cols ordered as [Att_1, FFN_1, Att_2, FFN_2, ..., Att_N, FFN_N].
        Also returns the original reps and atts matrices for compatibility.
        """
        model.set_use_module_grafting(False)
        model.set_use_scc_status(False)
    
        n_batch = len(dataloader)
        reps_similarity = None
        atts_similarity = None
        sub_similarity  = None
    
        progress_bar = tqdm(range(min(n_batch, max_iter)), desc="CKA Sublayer Evaluation")
    
        model.eval()
        for step, batch in enumerate(dataloader):
            if step == max_iter:
                break
    
            batch = {k: v.to(device) for k, v in batch.items()}
    
            with torch.no_grad():
                output = model(**batch)
    
            reps = output.hidden_states['hidden_states'][1:]
            atts = output.attentions['attention_head']
    
            if sub_similarity is None:
                reps_similarity = self._pairwise_helper(reps, only_cls_token=only_cls_token)
                atts_similarity = self._pairwise_helper(atts, is_att=True)
                sub_similarity  = self._interleaved_pairwise_helper(atts, reps, only_cls_token=only_cls_token)
            else:
                reps_similarity += self._pairwise_helper(reps, only_cls_token=only_cls_token)
                atts_similarity += self._pairwise_helper(atts, is_att=True)
                sub_similarity  += self._interleaved_pairwise_helper(atts, reps, only_cls_token=only_cls_token)
    
            progress_bar.update(1)
    
        num_steps = min(n_batch, max_iter)
        reps_similarity = reps_similarity / num_steps
        atts_similarity = atts_similarity / num_steps
        sub_similarity  = sub_similarity  / num_steps
    
        return reps_similarity, atts_similarity, sub_similarity