'''
We format the output like: 
<generated_keyphrase 1>, <generated_keyphrase 2>, ...<sep><ground_truth 1>, <ground_truth 2>, ... .
we write the output to a txt file, and then use the compute_metrics function to compute the metrics.
'''
import torch
from argparse import ArgumentParser, ArgumentTypeError
from filelock import FileLock
from accelerate import Accelerator
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from PIL import Image
from tqdm import tqdm
from metrics import compute_metrics
from torch.utils.data import Dataset, DataLoader
from llava.mm_utils import process_images
from PIL import Image
import json
from llava import conversation as conversation_lib
IMAGE_FILE = "path/to/datasets/CMKP_images/"
TXT_FILE = "path/to/outputs/"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')
    
def preprocess_v1(tokenizer, source):
    conv = conversation_lib.default_conversation.copy()
    conv.messages = []
    conv.append_message(conv.roles[0], source[0]["value"])
    conv.append_message(conv.roles[1], "")
    return tokenizer_image_token(conv.get_prompt(), tokenizer, return_tensors='pt')

def load_image(image_file):
    image_file = os.path.join(IMAGE_FILE,image_file.split('/')[-1])
    image = Image.open(image_file).convert('RGB')
    return image
class KpDataset(Dataset):
    def __init__(self, args, tokenizer, config, image_processor = None):
        self.data = json.load(open(args.data_path, "r"))[::args.stride]
        self.image_processor = image_processor
        self.args = args
        self.tokenizer = tokenizer
        self.model_config = config
    
    def __getitem__(self, idx):
        data = self.data[idx]
        if self.args.text_only:
            data['conversations'][0]['value'] = data['conversations'][0]['value'].replace("<image>", "")
        if self.args.image_only:
            data['conversations'][0]['value'] = data['conversations'][0]['value'].replace(data['text'], "")
        input_ids = preprocess_v1(self.tokenizer, data['conversations']).unsqueeze(0)
        label = data['conversations'][1]['value']
        image_size = None
        image = load_image(data['image'])
        image_tensor = process_images([image], self.image_processor, self.model_config)
        image_size = image.size
        item = {
            "idx":idx,
            "input_id":input_ids,
            "label":label,
            "image_tensor":image_tensor,
            "image_size":image_size,
        }
        return item
    
    def __len__(self):
        return len(self.data)

def custom_collate_fn(batch):
    if len(batch) == 1:
        return batch[0]
    else:
        raise NotImplementedError

def sort_by_id(file_path):
    with open(file_path, 'r') as f:
        data = f.read().split('\n')
    if data[-1] == "":
        data = data[:-1]
    d_list = {}
    for i in range(len(data)):
        d = data[i]
        ds = d.split("<sep>")
        if len(ds) < 3:
            continue
        idx = ds[0]
        d = "<sep>".join(ds[1:])
        d_list[int(idx)] = d
    with open(file_path, 'w') as f:
        for i in range(len(d_list)):
            f.write(d_list[i]+'\n')

import os
def is_accelerate_launch():
    return "ACCELERATE_DYNAMO_MODE" in os.environ

def evaluate(args):
    txt_file_path = os.path.join(TXT_FILE,args.txt_path+".txt")
    accelerator = None
    device = 'cuda' 
    is_acc = is_accelerate_launch()

    lock_file_path = os.path.join(TXT_FILE, "lock_file.lock")
    lock = FileLock(lock_file_path)
    if is_acc:
        accelerator = Accelerator()
        device = accelerator.device

    model_path = args.model_path
    model_base = args.model_base
    model_name = model_path
    if model_base is not None :
        if args.lora:
            model_name = "llava_lora" + get_model_name_from_path(model_path)
        else:
            model_name = "llava" + get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, False, False, device_map=device,device=device)#,use_flash_attn=True)

    dataset = KpDataset(args, tokenizer,model.config,image_processor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, num_workers=2)
    
    if is_acc:
        model, dataloader= accelerator.prepare(model, dataloader)
        if hasattr(model, "module"):
            model = model.module
    model.eval()
    progress_bar = tqdm(total=len(dataloader), desc="Processing")
    string_list = []
    for data in dataloader:
        idx = data['idx']
        input_ids = data['input_id'].to(model.device)
        label = data['label']
        image_tensor = data['image_tensor'].to(model.device, dtype=torch.float16)
        image_size = data['image_size']

        max_new_tokens = 40
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=args.do_sample,
                temperature=args.temperature,
                max_new_tokens=max_new_tokens,
                num_beams = args.beam,
                use_cache=True)
                
        label = [l.strip().lower() for l in label.split(",")]
        outputs = tokenizer.decode(output_ids[0],skip_special_tokens=True)
        string_list.append(str(idx)+"<sep>"+",".join(label)+"<sep>"+outputs+"")
        
        if args.print:
            print(outputs)
        if is_acc:
            if accelerator.is_local_main_process:
                progress_bar.update(1)
        else:
            progress_bar.update(1)

    if is_acc:
        accelerator.wait_for_everyone()
    with lock:
        with open(txt_file_path, 'a') as file:
            for s in string_list:
                file.write(s+"\n")
    if is_acc:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            sort_by_id(txt_file_path)
            compute_metrics(txt_file_path)
            print(f'save to {txt_file_path}')   
    else:
        sort_by_id(txt_file_path)
        compute_metrics(txt_file_path)
        print(f'save to {txt_file_path}')   

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--print", action='store_true',help="whether to print the output")
    parser.add_argument("--model-path", type=str, default="path/to/models/llava-v1.5-7b", help="path to the model")
    parser.add_argument("--model-base", type=str, default = None, help="path to base model")
    parser.add_argument("--data-path", type=str, default="path/to/datasets/test.json", help="path to the data")
    parser.add_argument("--beam",type=int ,default= 5)
    parser.add_argument("--temperature",type=float ,default= 0.5,help="temperature in inference")
    parser.add_argument("--stride",type=int ,default= 1,help="stride of data, used to reduce the size")
    parser.add_argument('--lora', type=str2bool, default=True, help='use when trained with LORA')
    parser.add_argument('--txt-path', type=str, default='test', help='txt path to save the output')
    parser.add_argument('--do-sample', type=str2bool, default=True, help='Whether to use sampling')
    parser.add_argument('--text-only', action='store_true', default=False, help='Whether to use text-only evaluation')
    parser.add_argument('--image-only', action='store_true', default=False, help='Whether to use image-only evaluation')
    args = parser.parse_args()
    print(args)
    evaluate(args)

# To speed up the eval process with multiple GPUs
# $ accelerate config
# accelerate launch evaluate.py --args