"""
Code adapated from https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py
Thanks to the authors of OpenCLIP
"""
import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import classification_report, balanced_accuracy_score


def zero_shot_classifier(model, tokenizer, classnames, templates, device):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.
    

    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer
    
    classnames: list of str
        name of classes
    
    templates: list of str
        templates to use.
    
    Returns
    -------
    
    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(c=classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.
    
    target: torch.Tensor
        shape (N,) where N is the number of examples. Class id of each example.
    
    topk: tuple
        which topk to compute, topk
    Returns
    -------
    
    list of top-k accuracies in the same order of `topk`
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().mean(0, keepdim=True).cpu().numpy()) for k in topk]


def run_classification(model, classifier, dataloader, device, amp=False):
    """
    Run zero-shot classifcation

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    classifier: torch.Tensor
        obtained from `zero_shot_classifier`
    
    dataloader: torch.utils.data.Dataloader 
        torch
    
    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    pred = []
    true = []
    nb = 0
    with torch.no_grad():
        for images, target in tqdm(dataloader):
            images = images.to(device)
            target = target.to(device)

            with autocast():
                # predict
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier
            
            true.append(target.cpu())
            pred.append(logits.cpu())

    pred = torch.cat(pred)
    true = torch.cat(true)
    return pred, true

def evaluate(model, dataloader, tokenizer, classnames, templates, device, amp=False, verbose=False):
    """
    Run zero-shot classification and evalue the metrics

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader

    tokenizer: text tokenizer

    classnames: list of str
        class names
    
    templates: list of str
        templates to use for zero-shot classification
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    verbose: whether to use verbose model

    Returns
    -------

    dict of classification metrics
    """
    classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device)
    logits, target = run_classification(model, classifier, dataloader, device, amp=amp)
    pred = logits.argmax(axis=1)
    # measure accuracy
    if len(dataloader.dataset.classes) >= 5:
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
    else:
        acc1, = accuracy(logits, target, topk=(1,))
        acc5 = float("nan") 
    mean_per_class_recall = balanced_accuracy_score(target, pred)
    if verbose:
        print(classification_report(target, pred, digits=3))
    return {"acc1": acc1, "acc5": acc5, "mean_per_class_recall": mean_per_class_recall}

