import torch
import clip
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


class ImageEmbddingLoss(torch.nn.Module):

    def __init__(self):
        super(ImageEmbddingLoss, self).__init__()
        self.model, _ = clip.load("ViT-B/32", device="cuda")
        self.transform = transforms.Compose([transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.cosloss = torch.nn.CosineEmbeddingLoss()
        self.str = '../datasets/templates.txt'
        with open(self.str, "r") as templates:
            text_prompt_templates = templates.readlines()
        self.text_prompts_templates = text_prompt_templates

    def get_delta_i(self, text_prompts):
        text_features = self._get_averaged_text_features(text_prompts)
        delta_t = text_features[0] - text_features[1]
        # delta_i = delta_t / torch.norm(delta_t)
        return delta_t

    def _get_averaged_text_features(self, text_prompts):
        with torch.no_grad():
            text_features_list = []
            for text_prompt in text_prompts:
                formatted_text_prompts = [template.format(text_prompt) for template in self.text_prompts_templates]  # format with class  （79）
                # print("formatted_text_prompts:", len(formatted_text_prompts))
                formatted_text_prompts = clip.tokenize(formatted_text_prompts).cuda()  # tokenize  (79 * 77)
                # print("formatted_text_prompts:", formatted_text_prompts.shape)
                text_embeddings = self.model.encode_text(formatted_text_prompts)  # embed with text encoder  (79 * 512)
                # print("text_embeddings:", text_embeddings.shape)
                # text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

                text_embedding = text_embeddings.mean(dim=0)
                # text_embedding /= text_embedding.norm()
                text_features_list.append(text_embedding)
            text_features = torch.stack(text_features_list, dim=1).cuda()
        return text_features.t()


    def forward(self, masked_generated=None, masked_img_tensor=None, text=None, delta=False, delta_i=None):

        if masked_generated is not None:
            masked_generated = self.face_pool(masked_generated)
            masked_generated_renormed = self.transform(masked_generated * 0.5 + 0.5)
            masked_generated_feature = self.model.encode_image(masked_generated_renormed)
            masked_generated_feature = masked_generated_feature / masked_generated_feature.norm(dim=-1, keepdim=True).float()
        # print(masked_generated_feature.shape)

        if masked_img_tensor is not None:
            masked_img_tensor = self.face_pool(masked_img_tensor)
            masked_img_tensor_renormed = self.transform(masked_img_tensor * 0.5 + 0.5)
            masked_img_tensor_feature = self.model.encode_image(masked_img_tensor_renormed)
            masked_img_tensor_feature = masked_img_tensor_feature / masked_img_tensor_feature.norm(dim=-1, keepdim=True).float()

        if delta is True:
            masked_generated_feature = masked_generated_feature - masked_img_tensor_feature
            masked_generated_feature = masked_generated_feature / masked_generated_feature.norm(dim=-1, keepdim=True).float()
            # print(delta_image.shape)

        if text is not None:
            text_ori = 'Photo'
            delta_text = self.get_delta_i([text, text_ori]).float()
            # text_tar = clip.tokenize(text).to(device)
            # text_ori = clip.tokenize(text_ori).to(device)
            # delta_text = self.model.encode_text(text_tar) - self.model.encode_text(text_ori)
            delta_text = delta_text.repeat(masked_generated_feature.shape[0], 1)
            # print(delta_text.shape)

        if delta_i is not None:
            delta_i = delta_i.repeat(masked_generated_feature.shape[0], 1)


        cos_target = torch.ones((masked_generated_feature.shape[0])).float().cuda()
        # print(cos_target.shape)
        similarity = self.cosloss(masked_generated_feature, delta_i, cos_target).unsqueeze(0).unsqueeze(0)
        return similarity
