import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from captum.attr import GuidedBackprop, GuidedGradCam


class HookFeatures:
    def __init__(self, module):
        self.feature_hook = module.register_forward_hook(self.feature_hook_fn)

    def feature_hook_fn(self, module, input, output):
        self.features = output.clone().detach()
        self.gradient_hook = output.register_hook(self.gradient_hook_fn)

    def gradient_hook_fn(self, grad):
        self.gradients = grad

    def close(self):
        self.feature_hook.remove()
        self.gradient_hook.remove()


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, action=None, trans=False):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.action = action
        self.trans = trans

    def forward(self, obs):
        if self.trans:
            return self.model(obs, self.action)[2]
        if self.action is None:
            return self.model(obs)[0]
        return self.model(obs, self.action)[0]


def compute_guided_backprop(obs, action, model, target, trans):
    model = ModelWrapper(model, action=action, trans=trans)
    gbp = GuidedBackprop(model)
    if target == None:
        attribution = gbp.attribute(obs)
    else:
        attribution = gbp.attribute(obs, target = target)
    return attribution

def compute_guided_gradcam(obs, action, model, target, trans):
    obs.requires_grad_()
    obs.retain_grad()
    model = ModelWrapper(model, action=action,  trans=trans)
    gbp = GuidedGradCam(model,layer=model.model.encoder.head_cnn.layers)
    if target == None:
        attribution = attribution = gbp.attribute(obs,attribute_to_layer_input=True)
    else:
        attribution = attribution = gbp.attribute(obs,attribute_to_layer_input=True, target = target)
    return attribution

def compute_vanilla_grad(critic_target, obs, action):
    obs.requires_grad_()
    obs.retain_grad()
    q, q2 = critic_target(obs, action.detach())
    q.sum().backward()
    return obs.grad


def compute_attribution(model, obs, action=None,method="guided_backprop", target = None, trans=False):
    if method == "guided_backprop":
        return compute_guided_backprop(obs, action, model, target, trans)
    if method == 'guided_gradcam':
        return compute_guided_gradcam(obs,action,model, target, trans)
    return compute_vanilla_grad(model, obs, action)


def compute_features_attribution(critic_target, obs, action):
    obs.requires_grad_()
    obs.retain_grad()
    hook = HookFeatures(critic_target.encoder)
    q, _ = critic_target(obs, action.detach())
    q.sum().backward()
    features_gardients = hook.gradients
    hook.close()
    return obs.grad, features_gardients


def compute_attribution_mask(obs_grad, quantile=0.95):
    mask = []
    for i in [0, 3, 6]:
        attributions = obs_grad[:, i : i + 3].abs().max(dim=1)[0]
        q = torch.quantile(attributions.flatten(1), quantile, 1)
        mask.append((attributions >= q[:, None, None]).unsqueeze(1).repeat(1, 3, 1, 1))
    return torch.cat(mask, dim=1)

def my_compute_attribution_mask(obs_grad, quantile=0.95):
    mask = []
    for i in [0, 3, 6]:
        attributions = obs_grad[:, i : i + 3].abs().max(dim=1)[0]
        # 获取尺寸
        n, h, w = attributions.size()
        flatten_attributions = attributions.flatten(1)
        max_values, _ = torch.max(flatten_attributions, dim=1, keepdim=True)
        is_zero_max = max_values == 0
        max_values = max_values.masked_fill(is_zero_max, 1)
        normalized_attributions = flatten_attributions / max_values
        temp_mask = normalized_attributions.reshape((n,h,w))
        mask.append(temp_mask.unsqueeze(1).repeat(1, 3, 1, 1))
    return torch.cat(mask, dim=1)

def my_compute_attribution_mask1(obs_grad, quantile=0.95):
    mask = []
    for i in [0, 3, 6]:
        attributions = obs_grad[:, i : i + 3].abs().max(dim=1)[0]
        # attributions_has_nan = torch.isnan(attributions).any().item()
        # print("attributions_has_nan", attributions_has_nan)
        # 获取尺寸
        n, h, w = attributions.size()
        max_values, _ = torch.max(attributions.flatten(1), dim=1, keepdim=True)
        is_zero_max = max_values == 0
        max_values = max_values.masked_fill(is_zero_max, 1)
        normalized_attributions = attributions.flatten(1) / max_values
        temp_mask = normalized_attributions.reshape((n,h,w))
        mask.append(temp_mask.unsqueeze(1).repeat(1, 3, 1, 1))
    return torch.cat(mask, dim=1)

def my_compute_attribution_mask2(obs_grad, quantile=0.95):
    mask = []
    for i in [0, 3, 6]:
        attributions = obs_grad[:, i : i + 3].abs().max(dim=1)[0]
        # 获取尺寸
        n, h, w = attributions.size()
        flatten_attributions = F.normalize(attributions.flatten(1), p=2, dim=1) 
        # print("flatten_attributions", flatten_attributions)
        max_values, _ = torch.max(flatten_attributions, dim=1, keepdim=True)
        # print("max_mean", max_values.mean())
        # print("max", max_values)
        normalized_attributions = flatten_attributions / max_values
        temp_mask = normalized_attributions.reshape((n,h,w))
        # print("ranks", ranks)
        mask.append(temp_mask.unsqueeze(1).repeat(1, 3, 1, 1))
    return torch.cat(mask, dim=1)

def make_obs_grid(obs, n=4):
    sample = []
    for i in range(n):
        for j in range(0, 9, 3):
            sample.append(obs[i, j : j + 3].unsqueeze(0))
    sample = torch.cat(sample, 0)
    return make_grid(sample, nrow=3) / 255.0


def make_attribution_pred_grid(attribution_pred, n=4):
    return make_grid(attribution_pred[:n], nrow=1)


def make_obs_grad_grid(obs_grad, n=4):
    sample = []
    for i in range(n):
        for j in range(0, 9, 3):
            channel_attribution, _ = torch.max(obs_grad[i, j : j + 3], dim=0)
            sample.append(channel_attribution[(None,) * 2] / channel_attribution.max())
    sample = torch.cat(sample, 0)
    q = torch.quantile(sample.flatten(1), 0.97, 1)
    sample[sample <= q[:, None, None, None]] = 0
    return make_grid(sample, nrow=3)



