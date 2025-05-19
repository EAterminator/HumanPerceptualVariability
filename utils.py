import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from models import DiT
from pipeWguidance import Pipe_guidance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def build_vit(path_model):
    from timm.models import  VisionTransformer
    vit = VisionTransformer(
        img_size=28, in_chans=1, num_classes=10,
        patch_size=4, embed_dim=128, depth=8, num_heads=4, mlp_ratio=4)
    vit.load_state_dict(torch.load(path_model))
    vit = vit.to(device)
    return vit

def build_vgg(path_model):
    from mnist_models import SmallVGG
    vgg = SmallVGG(num_classes=10)
    vgg.load_state_dict(torch.load(path_model))
    vgg = vgg.to(device)
    return vgg

def build_cornet(path_model):
    from mnist_models import CORnet_Z_cifar
    cornet_z = CORnet_Z_cifar(inchans=1)
    cornet_z.load_state_dict(torch.load(path_model))
    cornet_z = cornet_z.to(device)
    return cornet_z

def build_mlp(path_model):
    from mnist_models import MLP
    mlp = MLP()
    mlp.load_state_dict(torch.load(path_model))
    mlp = mlp.to(device)
    return mlp

def build_lrm(path_model):
    from mnist_models import LogisticRegressionModel
    lrm = LogisticRegressionModel(28*28, 10)
    lrm.load_state_dict(torch.load(path_model))
    lrm = lrm.to(device)
    return lrm

def divide_dataset(dataset, train_ratio=0.8):
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_valid = n - n_train
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [n_train, n_valid])
    return train_dataset, valid_dataset

def compare_tensors(tensor1, tensor2, get_same=False):
    # Ensure tensors are 1D
    if tensor1.dim() != 1 or tensor2.dim() != 1:
        raise ValueError("Both tensors must be 1D.")
    
    # Check if tensors have the same shape
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape.")
    # Compare tensors element-wise
    if get_same:
        indices = (tensor1 == tensor2).nonzero(as_tuple=True)[0]
    else:
        indices = (tensor1 != tensor2).nonzero(as_tuple=True)[0]
    
    if indices.numel() == 0:
        # Tensors are the same
        return True, None

    return False, indices

def merge_unique_indices(indices1, indices2):
    # Concatenate the two tensors
    if indices1==None and indices2 == None:
        return None
    elif indices1==None:
        combined_indices = indices2
    elif indices2==None:
        combined_indices = indices1
    else:
        combined_indices = torch.concat((indices1,indices2))
    # Get unique elements and sort them
    unique_indices = torch.unique(combined_indices, sorted=True)

    return unique_indices

def merge_same_indices(indices1, indices2):
    if indices1 is None and indices2 is None:
        return None
    elif indices1 is None or indices2 is None:
        return None
    else:
        mask = torch.isin(indices1, indices2)
        combined_indices = indices1[mask]
    return combined_indices

def show_image(image,nrow=10):
    """accept a tensor of shape (k,1,28,28)"""
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision
    img = torchvision.utils.make_grid(image, nrow=nrow)
    npimg =img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()
    
    
def get_loss_targeted_clf_ad_with_filter(clf1, clf2, y1, y2, filter_func):
    def loss_targeted_clf_ad_with_filter(x):
        logits_1 = clf1(x)
        logits_2 = clf2(x)
        score = filter_func(x)
        if not isinstance(score, torch.Tensor):
            score = torch.tensor(score, device=x.device, dtype=x.dtype)
        score_squared = (1-score) ** 2
        threshold = torch.tensor(0.5, device=x.device, dtype=x.dtype)
        penalty = torch.max(score_squared, threshold)
        loss = F.cross_entropy(logits_1, y1, reduction='mean') \
             + F.cross_entropy(logits_2, y2, reduction='mean') \
             + 10 * penalty.mean()
        return loss
    return loss_targeted_clf_ad_with_filter

def custom_cross_entropy(logits, targets):
    probabilities = F.softmax(logits, dim=1)
    log_probabilities = torch.log(probabilities + 1e-9)
    batch_loss = -torch.sum(targets * log_probabilities, dim=1)
    return torch.mean(batch_loss)

def get_loss_amb(clf, target):
    def loss_amb(x):
        logits = clf(x)
        loss = custom_cross_entropy(logits, target)
        return loss
    return loss_amb

def build_FilteringModel():
    from mnist_models import SmallVGG
    # small vgg + flatten
    vgg = nn.Sequential(
        SmallVGG(num_classes=1),
        nn.Flatten(start_dim=0)
    ).to(device)
    path_save = 'ckpts'
    vgg.load_state_dict(torch.load(os.path.join(path_save, 'model_filtering_Wmnist.pth')))
    return vgg

def generate_and_display(images_per_generation,folder_path,device='cuda'):# load all pth files in finetune_inlab with vgg in filename

    diffusion_prior = DiT(
        input_size=28, in_channels=1, class_dropout_prob=0.1,
        depth=4, num_heads=8, hidden_size=128, learn_sigma=False, num_classes=10
    )
    model_name = 'diffusion_DiT_mnist_uncond_snr'
    path = f'ckpts/{model_name}'
    diffusion_prior.load_state_dict(torch.load(f'{path}.pt'))
    
    digit_judge = build_FilteringModel()
    guide_pipe = Pipe_guidance(diffusion_prior, device=device)
    folder_path = folder_path
    pattern = os.path.join(folder_path, '*vgg*.pth')
    file_list = glob.glob(pattern)
    subject_names = []
    file_paths = []
    subs = []
    for file_path in sorted(file_list):
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        subject_name_with_ext = '_'.join(parts[2:])
        subject_name = os.path.splitext(subject_name_with_ext)[0]
        subject_names.append(subject_name)
        file_paths.append(file_path)
        model = build_vgg(file_path)
        subs.append(model)
    images_per_generation = images_per_generation
    y1_0 = torch.linspace(0,99,100,dtype=int).to(device)//10
    y2_0 = torch.linspace(0,99,100,dtype=int).to(device)%10
    y1_0 = torch.hstack([y1_0]*(images_per_generation//100))
    y2_0 = torch.hstack([y2_0]*(images_per_generation//100))
    repeated_indices = torch.where(y1_0 == y2_0)[0]
    y1 = y1_0
    y2 = y2_0
    for i in range(len(subs)):
        for j in range(i+1,len(subs)):
            print(f'generating for {subject_names[i]} and {subject_names[j]}')
            sample_times = 0
            indices = torch.arange(images_per_generation).to(device)
            image = torch.randn(images_per_generation,1,28,28).to(device)
            y1 = y1_0
            y2 = y2_0
            while True:
                resample_image = guide_pipe.generate_guidance(
                    loss_fn=get_loss_targeted_clf_ad_with_filter(subs[i],subs[j],y1,y2,digit_judge),
                    num_inference_steps=50,
                    guidance_scale=0.1,
                    N=indices.shape[0],
                    shape=(1,28,28),
                    num_resampling_steps=5
                )
                sample_times += 1
                image[indices] = resample_image
                clf1_judge_result = torch.argmax(subs[i](image),dim=1)
                clf2_judge_result = torch.argmax(subs[j](image),dim=1)
                
                max_value_1 = torch.max(F.softmax(subs[i](image),dim=1),dim=1)
                max_value_2 = torch.max(F.softmax(subs[j](image),dim=1),dim=1)
                
                satisfied_max1 = not torch.any(max_value_1.values < 0.5)
                satisfied_max2 = not torch.any(max_value_2.values < 0.5)
                indices_max1 = torch.where(max_value_1.values < 0.5)[0]
                indices_max2 = torch.where(max_value_2.values < 0.5)[0]
                
                scores = digit_judge(image)
                satisfied_judge = not torch.any(scores < 0.5)
                unsatisified_indices = torch.where(scores < 0.5)[0]
                
                satisfied1,indices1 = compare_tensors(y1_0,clf1_judge_result)
                satisfied2,indices2 = compare_tensors(y2_0,clf2_judge_result)
                if satisfied1 and satisfied2 and satisfied_max1 and satisfied_max2 and satisfied_judge:
                    image[repeated_indices] = torch.zeros_like(image[repeated_indices],device=device)
                    show_image(image)
                    plt.title(f'Controversial images for subject {subject_names[i]} and subject {subject_names[j]}')
                    break
                else:
                    indices = merge_unique_indices(indices1,indices2)
                    indices = merge_unique_indices(indices,indices_max1)
                    indices = merge_unique_indices(indices,indices_max2)
                    indices = merge_unique_indices(indices,unsatisified_indices)
                    y1 = y1_0[indices].to(device)
                    y2 = y2_0[indices].to(device)
                    if sample_times >= 10:
                        image[indices] = torch.zeros_like(image[indices],device=device)
                        image[repeated_indices] = torch.zeros_like(image[repeated_indices],device=device)
                        plt.title(f'Controversial images for subject {subject_names[i]} and subject {subject_names[j]}')
                        show_image(image)
                        break
    return image

def get_clf_names_and_list():
    vit = build_vit('ckpts/mnist_models/vit_mnist.pth').to(device)
    vgg = build_vgg('ckpts/mnist_models/small_vgg_mnist.pth').to(device)
    cornet = build_cornet('ckpts/mnist_models/cornet_z-mnist.pt').to(device)
    mlp = build_mlp('ckpts/mnist_models/mlp_mnist.pt').to(device)
    lrm = build_lrm('ckpts/mnist_models/lrm_mnist.pth').to(device)
    clf_list = [vit, vgg, cornet, mlp, lrm]
    clf_names = ['vit', 'vgg', 'cornet', 'mlp', 'lrm']
    return clf_list,clf_names

def uncertainty_guidance_generate(num_samples,clf,name,device='cuda'):
    diffusion_prior = DiT(
        input_size=28, in_channels=1, class_dropout_prob=0.1,
        depth=4, num_heads=8, hidden_size=128, learn_sigma=False, num_classes=10
    )
    model_name = 'diffusion_DiT_mnist_uncond_snr'
    path = f'ckpts/{model_name}'
    diffusion_prior.load_state_dict(torch.load(f'{path}.pt'))
    guide_pipe = Pipe_guidance(diffusion_prior, device=device)
    num_samples = num_samples
    target = torch.zeros(100, 10).to(device)
    for k in range(100):
        if k // 10 == k % 10:
            target[k, k // 10] = 1
        else:
            target[k, k // 10] = 0.5
            target[k, k % 10] = 0.5

    target = target.repeat(num_samples//100,1)
    indices = torch.arange(num_samples).to(device)
    print(f'generating with {name}')

    targeted_uncertainty_image = guide_pipe.generate_guidance(
        loss_fn = get_loss_amb(clf, target[indices,:]),
        num_inference_steps=50,
        guidance_scale=0.1,
        N=indices.shape[0],
        shape=(1,28,28),
        num_resampling_steps=5
    )
    plt.title(f'Images generated by {name}')
    show_image(targeted_uncertainty_image[:100])
    
    
def get_loss_targeted_clf_ad(clf1, clf2, y1, y2):
    def loss_clf_ad(x):
        logits_1 = clf1(x)
        logits_2 = clf2(x)
        return F.cross_entropy(logits_1, y1, reduction='mean') + F.cross_entropy(logits_2, y2, reduction='mean')
    return loss_clf_ad
    
def controversial_guidance_generate(num_samples,clf1,clf2,name1,name2,device='cuda'):
    diffusion_prior = DiT(
        input_size=28, in_channels=1, class_dropout_prob=0.1,
        depth=4, num_heads=8, hidden_size=128, learn_sigma=False, num_classes=10
    )
    model_name = 'diffusion_DiT_mnist_uncond_snr'
    path = f'ckpts/{model_name}'
    diffusion_prior.load_state_dict(torch.load(f'{path}.pt'))
    guide_pipe = Pipe_guidance(diffusion_prior, device=device)
    num_samples = num_samples
    y1_0 = torch.linspace(0,99,100,dtype=int).to(device)//10
    y2_0 = torch.linspace(0,99,100,dtype=int).to(device)%10
    y1_0 = torch.hstack([y1_0]*(num_samples//100))
    y2_0 = torch.hstack([y2_0]*(num_samples//100))
    indices = torch.arange(num_samples).to(device)
    print(f'generating with {name1} and {name2}')

    targeted_uncertainty_image = guide_pipe.generate_guidance(
        loss_fn=get_loss_targeted_clf_ad(clf1,clf2,y1_0,y2_0),
        num_inference_steps=50,
        guidance_scale=0.1,
        N=indices.shape[0],
        shape=(1,28,28),
        num_resampling_steps=5
    )
    plt.title(f'Images generated by {name1} and {name2}')
    show_image(targeted_uncertainty_image[:100])


