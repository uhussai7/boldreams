from lucent.optvis import render, param, objectives
from lucent.optvis.objectives import wrap_objective,handle_batch
import torch
from torch.nn import MSELoss

#the objectives functions here are all in the style of lucent

@wrap_objective()
def roi(roi, batch=None):
    """
    Maximize a single roi
    :param roi: Roi to maximize
    :param batch:
    :return: Objective
    """
    @handle_batch(batch)
    def inner(model):
        return -model(roi).mean()
    return inner

@wrap_objective()
def roi_ref_img(roi, ref_img,batch=None,sign=-1,gamma=1):
    """
    Maximize a single roi
    :param roi: Roi to maximize
    :param batch:
    :return: Objective
    """
    @handle_batch(batch)
    def inner(model):
        fmri,img=model(roi),model('ref_img')
        img=model('input')
        img_loss=((img-ref_img)*(img-ref_img)).mean()
        loss1=sign*fmri.mean()
        loss2=gamma*img_loss
        #print(loss1,loss2)
        return loss1+loss2
    return inner

@wrap_objective()
def roi_ref_img_roi_target(roi, ref_img,roi_target,batch=None,gamma=1):
    """
    Maximize a single roi
    :param roi: Roi to maximize
    :param batch:
    :return: Objective
    """
    @handle_batch(batch)
    def inner(model):
        fmri,img=model(roi),model('ref_img')
        img=model('input')
        img_loss=((img-ref_img)*(img-ref_img)).mean()
        loss1=fmri-roi_target
        loss1=loss1*loss1
        loss2=gamma*img_loss
        #print(loss1,loss2)
        return loss1.mean()+loss2
    return inner

# @wrap_objective()
# def roi_ref_img_targets(rois, ref_img,roi_targets,gamma=1,batch=None):
#     """
#     Maximize a single roi
#     :param roi: Roi to maximize
#     :param batch:
#     :return: Objective
#     """
#     @handle_batch(batch)
#     def inner(model):
#         fmri,img=model(roi),model('ref_img') #evaluate the model
#         fmri_roi_means = torch.stack([model(r).mean() for r in rois])
#         loss_fmri = roi_targets - fmri_roi_means
#         loss_fmri = (loss_fmri * loss_fmri).mean()
#         img_loss=((img-ref_img)*(img-ref_img)).mean()
#         loss2=gamma*img_loss
#         #print(loss1,loss2)
#         return loss_fmri+loss2
#     return inner

# @wrap_objective()
# def roi_ref_img_mean_targets(rois, ref_img,roi_targets,gamma=1,batch=None):
#     """
#     Maximize a single roi
#     :param roi: Roi to maximize
#     :param batch:
#     :return: Objective
#     """
#     @handle_batch(batch)
#     def inner(model):
#         fmri,img=model(roi),model('ref_img') #evaluate the model
#         fmri_roi_means = torch.stack([model(r).mean() for r in rois])
#         loss_fmri = roi_targets - fmri_roi_means
#         loss_fmri = (loss_fmri * loss_fmri).mean()
#         img_loss=((img-ref_img)*(img-ref_img)).mean()
#         loss2=gamma*img_loss
#         #print(loss1,loss2)
#         return loss_fmri+loss2
#     return inner
#
# @wrap_objective()
# def roi_ref_img_targets(rois, ref_img,roi_targets,gamma=1,batch=None,device='cuda'):
#     """
#     Maximize a single roi
#     :param roi: Roi to maximize
#     :param batch:
#     :return: Objective
#     """
#     @handle_batch(batch)
#     def inner(model):
#         for roi in rois:
#             #fmri_roi_means = torch.stack([model(roi) for r in rois])
#             losst=roi_targets[roi].to(device)
#         loss_fmri = roi_targets - fmri_roi_means
#         loss_fmri = (loss_fmri * loss_fmri).mean()
#         img_loss=((img-ref_img)*(img-ref_img)).mean()
#         loss2=gamma*img_loss
#         #print(loss1,loss2)
#         return loss_fmri+loss2
#     return inner

@wrap_objective()
def roi_targets(targets, batch=None,device='cuda'):
    @handle_batch(batch)
    def inner(model):
        loss=0
        for key in targets.keys():
            losst=targets[key].to(device)-model('roi_'+key)
            losst=losst*losst
            loss+=losst.mean()
        #print(loss)
        return loss
    return inner

@wrap_objective()
def roi_max_min(max_rois,min_rois,c=1,batch=None):
    """
    Maximize selected rois while minimizing others
    :param max_rois: list of max rois
    :param min_rois: list of min rois
    :param c: min/max mixing constant
    :param batch:
    :return: Objective
    """
    @handle_batch(batch)
    def inner(model):
        fmri_max_roi_mean=torch.cat([model(r).mean(0) for r in max_rois]).mean()
        fmri_min_roi_mean=torch.cat([model(r).mean(0) for r in min_rois]).mean()
        return -fmri_max_roi_mean + c*fmri_min_roi_mean
    return inner

@wrap_objective()
def roi_mean_target(rois,roi_targets,weight=None,batch=None):
    """
    Optimize towards mean target activations
    :param rois: List of rois
    :param roi_targets: Target mean activation for each roi
    :param weights: weights for each target
    :param batch:
    :return: Objective
    """
    @handle_batch(batch)
    def inner(model):
        fmri_roi_means=torch.stack([model(r).mean() for r in rois])
        loss=roi_targets-fmri_roi_means
        if weight is None:
            return (loss*loss).mean()
        else:
            return (loss*loss*weight).mean()
    return inner


@wrap_objective()
def diversity(roi):
    """
    Encourage diversity between each image in batch
    :param roi: Roi
    :return: Objective
    """
    def inner(model):
        fmri = model(roi)
        Nb, Nv= fmri.shape #Nbatch and Nvoxels
        out = -sum([ sum([ (fmri[i]*fmri[j]).sum() for j in range(Nb) if j != i])for i in range(Nb)]) / Nb
        return out
    return inner

@wrap_objective()
def clip_img_features(a=0.01):
    def inner(model):
        tt=model('model')[0][0].mean()
        ee=model('encoder')['ffa'].mean()
        return -(a*tt + ee)
    return inner