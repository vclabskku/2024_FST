import torch
import torch.nn as nn
from functools import partial

def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(x_tokens_list[-2][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()

def create_linear_input_concat(x_tokens_list1, x_tokens_list2, use_n_blocks, use_avgpool):
    # resize 
    intermediate_output1 = x_tokens_list1[-use_n_blocks:]
    output1 = torch.cat([class_token for _, class_token in intermediate_output1], dim=-1)
    # crop
    intermediate_output2 = x_tokens_list2[-use_n_blocks:]
    output2 = torch.cat([class_token for _, class_token in intermediate_output2], dim=-1)
    output = torch.cat([output1, output2], dim=-1)

    if use_avgpool:
        fea = torch.cat(
            (torch.mean(intermediate_output1[-1][0], dim=1), torch.mean(intermediate_output2[-1][0], dim=1))
            , dim=-1)
        output = torch.cat(
            (
                output,
                fea,  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()

class LinearClassifier(nn.Module):
    def __init__(self, out_dim, use_n_blocks, use_avgpool, concat, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.concat = concat
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        if self.concat:
            output = create_linear_input_concat(x_tokens_list[0], x_tokens_list[1], self.use_n_blocks, self.use_avgpool)
        else:
            output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)
    
class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)

def scale_lr(learning_rates, batch_size):
    return learning_rates * batch_size / 256.0

def setup_linear_classifiers(sample_output, n_last_blocks_list, learning_rates, batch_size, concat, num_classes=1000):
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []

    for n in n_last_blocks_list:
        for avgpool in [False, True]:
            for _lr in learning_rates:
                lr = scale_lr(_lr, batch_size)
                if concat:
                    out_dim = create_linear_input_concat(sample_output, sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                else:
                    out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                linear_classifier = LinearClassifier(
                    out_dim, use_n_blocks=n, use_avgpool=avgpool, concat=concat, num_classes=num_classes
                )
                linear_classifier = linear_classifier.cuda()
                linear_classifiers_dict[
                    f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")
                ] = linear_classifier
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)

    return linear_classifiers, optim_param_groups

class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()            
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.no_grad():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features
    
    
class Create_DINO(nn.Module):
    def __init__(self, model_name, sample, batch_size, concat, num_classes):
        super().__init__()
        model = torch.hub.load('facebookresearch/dinov2', model_name)
        autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)

        self.learning_rates = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
        self.n_last_blocks_list = [1, 4]
        self.concat = concat

        self.feature_model = ModelWithIntermediateLayers(model, max(self.n_last_blocks_list), autocast_ctx)
        sample_output = self.feature_model(sample)

        self.linear_classifiers, self.optim_param_groups = setup_linear_classifiers(
            sample_output, self.n_last_blocks_list, self.learning_rates, batch_size, self.concat, num_classes
        )

    def forward(self, images):
        if self.concat:
            fea1 = self.feature_model(images[0].cuda())
            fea2 = self.feature_model(images[1].cuda())
            outputs = self.linear_classifiers([fea1, fea2])
        else:
            features = self.feature_model(images)
            outputs = self.linear_classifiers(features)
        return outputs
