import argparse
from transformers import AutoTokenizer
import torch
from chatbind.conversation import conv_templates, SeparatorStyle
from chatbind.utils import disable_torch_init
from transformers import CLIPImageProcessor
from chatbind.model import *
from chatbind.model.utils import KeywordsStoppingCriteria
from chatbind import *

import os
import requests
from PIL import Image
from io import BytesIO


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "mpt" in model_name.lower():
        model = LlavaMPTForCausalLM.from_pretrained(model_name, use_cache=True).cuda()
    else:
        model = LlavaLlamaImageBindSelectAllForCausalLM.from_pretrained(model_name, use_cache=True).cuda()
    if args.modality_mode == 'image':
        data_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower)
    elif args.modality_mode == 'audio':
        data_processor = load_and_transform_audio_data
    elif args.modality_mode == 'video':
        data_processor = load_and_transform_video_data
    else:
        raise ValueError

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    setattr(model.get_model(), 'repeat_feature_dim', args.repeat_feature_dim)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]
    vision_tower.to(device='cuda')
    vision_config = vision_tower.config

    if args.pretrain_mm_mlp_adapter:
        print(f'load {args.pretrain_mm_mlp_adapter}')
        mm_projector_weights = torch.load(args.pretrain_mm_mlp_adapter, map_location='cpu')
        msg = model.get_model().mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})
        print(msg)

    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    image_token_len = args.repeat_feature_dim

    qs = args.query
    if mm_use_im_start_end:
        qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
    else:
        qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    if "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt_multimodal"
    else:
        conv_mode = "multimodal"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    print(args.conv_mode)
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])

    if args.modality_mode == 'image':
        image = load_image(args.input_file)
        image_tensor = data_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).cuda()
        data_list = {ModalityType.VISION: image_tensor}
    elif args.modality_mode == 'audio':
        image_tensor = data_processor([args.input_file, ], 'cuda')
        data_list = {ModalityType.AUDIO: image_tensor}
    elif args.modality_mode == 'video':
        image_tensor = data_processor([args.input_file, ], 'cuda')
        data_list = {ModalityType.VISION: image_tensor}
    else:
        raise ValueError

    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            # images=image_tensor.unsqueeze(0).cuda(),
            images=data_list,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--modality-mode", type=str, default='image')
    parser.add_argument('--pretrain-mm-mlp-adapter', type=str, default=None)
    parser.add_argument('--repeat-feature-dim', type=int, default=1)
    args = parser.parse_args()

    eval_model(args)