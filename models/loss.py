from operator import index
import torch
import torch.nn.functional as F
import pdb
import copy


def cal_nll_loss(logit, idx, mask, weights=None):
    eps = 0.1
    acc = (logit.max(dim=-1)[1]==idx).float()
    mean_acc = (acc * mask).sum() / mask.sum()
    
    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -logit.sum(dim=-1)
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        nll_loss = (nll_loss * weights).sum(dim=-1)

    return nll_loss.contiguous(), mean_acc


def mlm_loss(words_logit, words_id, words_mask, txt_concepts, pos_words_logit, ref_words_logit, num_props, training_rec_only=False, **kwargs):
    

    if pos_words_logit != None and not training_rec_only:
        final_loss = 0
        rec_loss, txt_acc = cal_nll_loss(words_logit, words_id, words_mask)
        bsz, num_concepts, _ = txt_concepts.shape
        rec_loss = rec_loss.mean() * kwargs["alpha_2"]
        final_loss += rec_loss

        words_mask1 = words_mask.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
        words_id1 = words_id.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
        nll_loss, vid_acc = cal_nll_loss(pos_words_logit, words_id1, words_mask1)
        min_nll_loss = nll_loss.view(bsz, num_props).min(dim=-1)[0].mean() * kwargs["alpha_1"]
        final_loss += min_nll_loss

        loss_dict = {
            '\nReconstruction Loss:': final_loss.item(),
            '(1) txt_rec_loss:': rec_loss.item(),
            '(2) vid_rec_loss:': min_nll_loss.item() if min_nll_loss != None else 0,
        }
        return final_loss, loss_dict, txt_acc, vid_acc
    else:
        final_loss = 0
        nll_loss, vid_acc = cal_nll_loss(ref_words_logit, words_id, words_mask)
        rec_loss = nll_loss.mean()
        final_loss += rec_loss
        bsz, num_concepts, _ = txt_concepts.shape

        words_mask1 = words_mask.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
        words_id1 = words_id.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
        nll_loss, vid_acc = cal_nll_loss(pos_words_logit, words_id1, words_mask1)
        min_nll_loss = nll_loss.view(bsz, num_props).min(dim=-1)[0].mean()
        final_loss += min_nll_loss
        loss_dict = {
            '\nReconstruction Loss:': final_loss.item(),
            '(1) ref_rec_loss:': rec_loss.item(),
            '(2) prop_rec_loss:': min_nll_loss.item(),
        }
        return final_loss, loss_dict, vid_acc


    



def div_loss(pos_vid_concepts, txt_concepts, gauss_weight, num_concepts=10, num_props=5, **kwargs):
    loss = 0
    bsz = txt_concepts.shape[0]

    gauss_weight = gauss_weight[:, :-1]
    gauss_weight = gauss_weight.reshape(bsz, num_props, -1)
    if gauss_weight != None:
        gauss_weight = gauss_weight / gauss_weight.sum(dim=-1, keepdim=True)
        target = torch.eye(num_props).unsqueeze(0).cuda() * (kwargs["delta"])
        source = torch.matmul(gauss_weight, gauss_weight.transpose(1, 2))
        anchor_div_loss = (torch.norm(target - source, dim=(1, 2))**2).mean() * (kwargs["epsilon"])
        loss += anchor_div_loss

    loss_dict = {
            '\nDiversity Loss:': loss.item(),
            '(1) anchor_div_loss': anchor_div_loss.item(),
        }

    return loss, loss_dict


def cma_loss(pos_vid_concepts, txt_concepts, neg_vid_concepts_1, pos_words_logit, ref_concept, words_id, words_mask, use_ref_words_sam=True, num_concepts=4, num_props=5, **kwargs):

    pos_vid_concepts = pos_vid_concepts.detach()
    neg_vid_concepts_1 = neg_vid_concepts_1.detach()

    
    loss = 0
    _, num_concepts, D = pos_vid_concepts.shape

    bsz = pos_words_logit.size(0) // num_props
    words_mask1 = words_mask.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    nll_loss, acc = cal_nll_loss(pos_words_logit, words_id1, words_mask1)
    _, idx = nll_loss.view(bsz, num_props).min(dim=-1)

    idx = idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_concepts, D)
    pos_vid_concept = pos_vid_concepts.view(bsz, num_props, num_concepts, -1).gather(index=idx, dim=1).squeeze(dim=1)
    neg_vid_concept_1 = torch.gather(neg_vid_concepts_1.view(bsz, num_props, num_concepts, -1), index=idx, dim=1).squeeze(dim=1)

    pos_samilarity = torch.matmul(F.normalize(pos_vid_concept,dim=-1, p=2), F.normalize(txt_concepts,dim=-1, p=2).transpose(1, 2))
    pos_samilarity = torch.diagonal(pos_samilarity, dim1=-1, dim2=-2)

    pos_mse_loss = F.mse_loss(pos_vid_concept, txt_concepts, reduction='none').mean()
    loss += pos_mse_loss

    if neg_vid_concepts_1 != None:
        neg_samilarity_1 = torch.matmul(F.normalize(neg_vid_concept_1,dim=-1, p=2), F.normalize(txt_concepts,dim=-1, p=2).transpose(1, 2))
        neg_samilarity_1 = torch.diagonal(neg_samilarity_1, dim1=-1, dim2=-2)

        tmp_0 = torch.zeros_like(pos_samilarity).cuda()
        tmp_0.requires_grad = False
        samilarity_loss_1 = torch.max(neg_samilarity_1 - pos_samilarity + kwargs["margin_4"], tmp_0).sum(dim=-1).mean()
        loss = loss + samilarity_loss_1

    loss_dict = {
        '\nMultimodal Concept Loss:': loss.item(),
        '(1) pos_mse_loss': pos_mse_loss.item(),
        '(2) samilarity_loss_1': samilarity_loss_1.item() if neg_vid_concepts_1 != None else 0, 
    }
    

    return loss, loss_dict



def ivc_loss(pos_words_logit, words_id, words_mask, num_props, neg_words_logit_1=None, ref_words_logit=None, use_ref_words_rec=True, **kwargs):
    bsz = pos_words_logit.shape[0]//num_props

    words_mask1 = words_mask.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

    nll_loss, acc = cal_nll_loss(pos_words_logit, words_id1, words_mask1)
    min_nll_loss, idx = nll_loss.view(bsz, num_props).min(dim=-1)

    rank_loss = 0

    
    if neg_words_logit_1 is not None:
        neg_nll_loss_1, neg_acc_1 = cal_nll_loss(neg_words_logit_1, words_id1, words_mask1)
        neg_nll_loss_1 = torch.gather(neg_nll_loss_1.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        neg_loss_1 = torch.max(min_nll_loss - neg_nll_loss_1 + kwargs["margin_2"], tmp_0)
        rank_loss = rank_loss + neg_loss_1.mean()
    


    loss = kwargs['alpha_1'] * rank_loss

    return loss, {
        '\nIntra-Video Loss': loss.item(),
        '(1) hinge_loss_neg1': neg_loss_1.mean().item() if neg_words_logit_1 is not None else 0.0,
    }


def cvc_loss(ref_concept, txt_concepts, pos_vid_concepts, pos_words_logit, words_id, words_mask, num_props=5, num_concepts=1, use_query_triplet=False, **kwargs):
    bsz = ref_concept.shape[0]
    D = ref_concept.shape[-1]

    loss = 0


    words_mask1 = words_mask.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    nll_loss, acc = cal_nll_loss(pos_words_logit, words_id1, words_mask1)
    _, idx = nll_loss.view(bsz, num_props).min(dim=-1)

    idx = idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_concepts, D)
    pos_vid_concept = pos_vid_concepts.view(bsz, num_props, num_concepts, -1).gather(index=idx, dim=1).squeeze(dim=1) # (bsz, num_concepts, -1)

    

    pos_vid_concept = F.normalize(pos_vid_concept[:,0,:],dim=-1, p=2)
    ref_concept = F.normalize(ref_concept[:,0,:],dim=-1, p=2)
    txt_concept = F.normalize(txt_concepts[:,0,:],dim=-1, p=2)

    x = torch.matmul(ref_concept, txt_concept.t())
    x = x.view(bsz, bsz, -1)
    nominator = x * torch.eye(x.shape[0]).cuda()[:, :, None] # equal to unsqueeze(-1)
    nominator = nominator.sum(dim=1)
    nominator = torch.logsumexp(nominator, dim=1)
    denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
    denominator = torch.logsumexp(denominator, dim=1)
    ref_nce_loss = (denominator - nominator).mean() * kwargs["cvc_scaler_1"]
    loss += ref_nce_loss

    x = torch.matmul(pos_vid_concept, txt_concept.t())
    x = x.view(bsz, bsz, -1)
    nominator = x * torch.eye(x.shape[0]).cuda()[:, :, None] # equal to unsqueeze(-1)
    nominator = nominator.sum(dim=1)
    nominator = torch.logsumexp(nominator, dim=1)
    denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
    denominator = torch.logsumexp(denominator, dim=1)
    pos_nce_loss = (denominator - nominator).mean() * kwargs["cvc_scaler_2"]
    loss += pos_nce_loss

    x = torch.matmul(ref_concept, txt_concept.t())
    vid_triplet_loss = get_triplet_loss(x, margin=kwargs["margin_3"])
    if use_query_triplet:
        x = torch.matmul(txt_concept, ref_concept.t())
        txt_ref_triplet_loss = get_triplet_loss(x, margin=kwargs["margin_3"])
        vid_triplet_loss += txt_ref_triplet_loss
    loss += vid_triplet_loss
    
    x = torch.matmul(pos_vid_concept, txt_concept.t())
    prop_triplet_loss = get_triplet_loss(x, margin=kwargs["margin_3"])
    if use_query_triplet:
        x = torch.matmul(txt_concept, pos_vid_concept.t())
        txt_prop_triplet_loss = get_triplet_loss(x, margin=kwargs["margin_3"])
        prop_triplet_loss += txt_prop_triplet_loss
    loss += prop_triplet_loss



    return loss, {
        '\nCross-Video Loss': loss.item(),
        '(1) ref_nce_loss': ref_nce_loss.item() if ref_nce_loss is not None else 0.0,
        '(2) pos_nce_loss': pos_nce_loss.item() if pos_nce_loss is not None else 0.0,
        '(3) vid_triplet_loss': vid_triplet_loss.item() if vid_triplet_loss is not None else 0.0,
        '(4) prop_triplet_loss': prop_triplet_loss.item() if prop_triplet_loss is not None else 0.0,

    }



def get_triplet_loss(query_context_scores, margin=0.1):
    """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
    Args:
        query_context_scores: (N, N), cosine similarity [-1, 1],
            Each row contains the scores between the query to each of the videos inside the batch.
    """

    bsz = len(query_context_scores)

    diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
    pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
    query_context_scores_masked = copy.deepcopy(query_context_scores.data)
    # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
    query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
    pos_query_neg_context_scores = get_neg_scores(query_context_scores, query_context_scores_masked)
    neg_query_pos_context_scores = get_neg_scores(query_context_scores.transpose(0, 1), query_context_scores_masked.transpose(0, 1))
    loss_neg_ctx = get_ranking_loss(pos_scores, pos_query_neg_context_scores, margin)
    loss_neg_q = get_ranking_loss(pos_scores, neg_query_pos_context_scores, margin)
    return loss_neg_ctx + loss_neg_q

def get_neg_scores(scores, scores_masked):
    """
    scores: (N, N), cosine similarity [-1, 1],
        Each row are scores: query --> all videos. Transposed version: video --> all queries.
    scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
        are masked with a large value.
    """

    bsz = len(scores)
    batch_indices = torch.arange(bsz).to(scores.device)
    _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)
    sample_min_idx = 1  # skip the masked positive
    sample_max_idx = bsz

    # sample_max_idx = 2
    # (N, )
    sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx,
                                                                                   size=(bsz,)).to(scores.device)]
    sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N, )
    return sampled_neg_scores

def get_ranking_loss(pos_score, neg_score, margin=0.1):
    """ Note here we encourage positive scores to be larger than negative scores.
    Args:
        pos_score: (N, ), torch.float32
        neg_score: (N, ), torch.float32
    """
    return torch.clamp(margin + neg_score - pos_score, min=0).sum() / len(pos_score)