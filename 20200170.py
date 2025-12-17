import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import numpy as np
from PIL import Image
import json
from datasets import load_dataset
import copy
import os
import torch.nn.functional as F
import argparse
import random
from transformers import GPT2LMHeadModel
import math



# Constants
CIFAR_BATCH_SIZE = 128
LM_BATCH_SIZE = 32
VL_BATCH_SIZE = 16
MAX_LENGTH = 128
HIDDEN_SIZE = 768
NUM_EPOCHS = 1
IMG_PATCH = '<img>'
NUM_IMG_TOKEN = 32
VLM_MAX_LENGTH = 32
PAD_TOKEN_ID = 50256

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# CIFAR-10 Dataset and DataLoader
def get_cifar10_loaders():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=CIFAR_BATCH_SIZE,
                           shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=CIFAR_BATCH_SIZE,
                          shuffle=False, num_workers=2)
    
    return trainloader, testloader

# ELI5 Dataset
class ELI5Dataset(Dataset):
    def __init__(self,tokenizer, MAX_POSITION_EMBEDDINGS, data_type):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.block_size = MAX_POSITION_EMBEDDINGS
        
        if data_type == "train":
            data = load_dataset("eli5_category", split="train[:3000]", trust_remote_code=True)
            data = data.select(range(1000))
        elif data_type == "valid":
            data = load_dataset("eli5_category", split="validation1[:2000]", trust_remote_code=True)
        elif data_type == "test":
            data = load_dataset("eli5_category", split="test[:20]", trust_remote_code=True)

        data = data.flatten() 
        data = data.map(self.preprocess_function, batched=True,num_proc=8,remove_columns=data.column_names)
        data = data.map(self.group_texts, batched=True, num_proc=8)
        result =[]
        for i in data:
            result.append(i['input_ids'])
        self.final_data = torch.tensor(result).to(torch.int64)
        
    def preprocess_function(self, examples):
        return self.tokenizer([" ".join(x) for x in examples["answers.text"]])
    
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]]) 
        if total_length >= (self.block_size-2):
            total_length = (total_length // (self.block_size-2)) * (self.block_size-2)
        result = {
            k: [[self.tokenizer.bos_token_id]+t[i : i + self.block_size-2]+[self.tokenizer.eos_token_id] for i in range(0, total_length, self.block_size-2)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
        
    def __len__(self):
        return len(self.final_data)
    
    def __getitem__(self, idx):
        return self.final_data[idx]

# LLaVA Dataset
def transform_fn(is_train):
    if is_train:
        return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    else:
        return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Do not change
class LLaVADataset(Dataset):
    def __init__(self, json_file, img_path, tokenizer, is_train):
        super().__init__()

        self.transform = transform_fn(is_train)

        self.json_file = json_file

        self.tokenizer = tokenizer
        self.img_path = img_path

        self.ignore_idx = -100
        self.begin_signal = tokenizer.bos_token
        self.end_signal = tokenizer.eos_token

        with open(self.json_file) as json_file:
            data = json.load(json_file)

        if is_train:
            data = data[:1000]
        else:
            data = data[1000:]

        self.data = data

    def preprocess(self, conversation):
        question = self.begin_signal + "human: " + conversation[0]['value'] + self.end_signal
        answer = self.begin_signal + "assistant: " + conversation[1]['value'] + self.end_signal

        tokenized_q = self.tokenizer(question, return_tensors="pt")

        combined_qa = question + answer
        tokenized_qa = self.tokenizer(combined_qa, padding="max_length", truncation=True,
                                      max_length=VLM_MAX_LENGTH, return_tensors="pt")

        input_ids = tokenized_qa.input_ids[0]
        label = copy.deepcopy(input_ids)
        len_of_q = len(tokenized_q.input_ids[0])
        label[:len_of_q] = self.ignore_idx

        len_of_pad = tokenized_qa.input_ids.eq(self.tokenizer.pad_token_id).sum().item()
        label[-len_of_pad:] = self.ignore_idx

        return input_ids, label
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        meta = self.data[idx]

        image_id = meta['image']
        image = Image.open(os.path.join(self.img_path, image_id)).convert('RGB')
        image = self.transform(image)

        conversation = meta['conversation']
        input_id, label = self.preprocess(conversation)

        return dict(image=image, input_ids=input_id, label=label)
    




def get_image_encoder():
    m = torchvision.models.resnet18()
    m.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = torch.nn.Identity()
    return m
    

class VisionClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, feat_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # resnet forward: returns logits via encoder.fc normally.
        # We'll bypass encoder.fc and use our own.
        # For torchvision resnet: feature before fc is obtained via avgpool then flatten.
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)  # [B, 512]
        return self.fc(x), x

def train_vision_classifier(model, train_dataloader, test_dataloader, optimizer, epoch, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for i in range(epoch):
        total_loss = 0.0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / len(train_dataloader.dataset)
        print(f"Epoch [{i+1}/{epoch}], Loss: {avg_loss:.4f}")
        test_vision_classifier(model, test_dataloader, device)
        
    return avg_loss

@torch.no_grad()
def test_vision_classifier(model, dataloader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs, _ = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    return accuracy

def train_text_decoder(model, dataloader, optimizer, epoch, device='cuda'):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    model.to(device)
    model.train()
    for i in range(epoch):
        total_loss = 0.0
        for batch in dataloader:
            inputs = batch.to(device)
            labels = inputs.clone()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs.logits
            # import ipdb; ipdb.set_trace()
            loss = criterion(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
            # loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch [{i+1}/{epoch}], Loss: {avg_loss:.4f}")
        
    return avg_loss

class PrefixProjector(nn.Module):
    def __init__(self, in_dim: int, hidden: int, prefix_len: int):
        super().__init__()
        self.prefix_len = prefix_len
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden * prefix_len),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, in_dim]
        x = self.proj(feat)  # [B, prefix_len*hidden]
        return x.view(x.size(0), self.prefix_len, -1)  # [B, P, hidden]



class VisionLanguageMachine(nn.Module):
    def __init__(self, vision_encoder: nn.Module, text_decoder: nn.Module):
        super().__init__()
        self.prefix_len = NUM_IMG_TOKEN
        
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.projector = PrefixProjector(in_dim=512, hidden=HIDDEN_SIZE, prefix_len=self.prefix_len)

    def forward(self, images, input_ids, labels=None):
        B = input_ids.size(0)
        
        # Extract visual features
        _, visual_feats = self.vision_encoder(images)
        prefix = self.projector(visual_feats)  # [B, P, HIDDEN_SIZE]
        
        text_mask = (input_ids != PAD_TOKEN_ID).long()
        tok_emb = self.text_decoder.get_input_embeddings()(input_ids)  # [B, T, H]
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
        
        prefix_mask = torch.ones((B, self.prefix_len), device=text_mask.device, dtype=text_mask.dtype)
        attention_mask = torch.cat([prefix_mask, text_mask], dim=1)
        
        prefix_labels = torch.full((B, self.prefix_len), -100, device=labels.device, dtype=labels.dtype)
        lab = torch.cat([prefix_labels, labels], dim=1)
        
        out = self.text_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=lab,
        )
        
        logits = out.logits[:, self.prefix_len:, :].contiguous()
        
        return logits
    
def train_vlm(model, dataloader, optimizer, epoch, device='cuda'):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    model.to(device)
    model.train()
    for i in range(epoch):
        total_loss = 0.0
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(images, input_ids, labels=labels)
            # loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = criterion(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch [{i+1}/{epoch}], Loss: {avg_loss:.4f}")
        
    return avg_loss


# capture perplexity on test set
@torch.no_grad()
def test_vlm(model, dataloader, device='cuda'):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    total_nll, total_tok = 0.0, 0
    
    logits_total = []
    for batch in dataloader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(images, input_ids, labels=labels)
        nll = criterion(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
        ).item()
        
        tok = (labels[:, 1:] != -100).sum().item()
        total_nll += nll
        total_tok += tok
        
        logits_total.append(logits.cpu().numpy().astype(np.float16))
        
        
        # total_loss += loss.item() * images.size(0)
    avg_nll = total_nll / total_tok
    ppl = math.exp(avg_nll)
    print(f'Test Loss: {avg_nll:.4f}, Perplexity: {ppl:.4f}')
    logits_total = np.concatenate(logits_total, axis=0)
    return avg_nll, logits_total
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str)
    parser.add_argument('--image_folder_path', type=str)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(IMG_PATCH, special_tokens=True)
    
    # cifar10
    cifar_trainloader, cifar_testloader = get_cifar10_loaders()
    
    # eli5
    eli5_dataset = ELI5Dataset(tokenizer, MAX_LENGTH, 'train')
    eli5_loader = DataLoader(eli5_dataset, batch_size=LM_BATCH_SIZE, shuffle=True)
    
    # llava train
    llava_dataset = LLaVADataset(args.json_path, args.image_folder_path, tokenizer, is_train=True)
    llava_loader = DataLoader(llava_dataset, batch_size=VL_BATCH_SIZE, shuffle=True)
    
    # llava test
    test_llava_dataset = LLaVADataset(args.json_path, args.image_folder_path, tokenizer, is_train=False)
    test_llava_loader = DataLoader(test_llava_dataset, batch_size=VL_BATCH_SIZE, shuffle=False)
    
    
    vision_encoder = get_image_encoder()
    vision_classifier = VisionClassifier(vision_encoder)
    vision_optimizer = torch.optim.Adam(vision_classifier.parameters(), lr=0.001)
    train_vision_classifier(vision_classifier, cifar_trainloader, cifar_testloader, vision_optimizer, 10, device)
    
    decoder = GPT2LMHeadModel.from_pretrained('gpt2')
    text_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
    train_text_decoder(decoder, eli5_loader, text_optimizer, 3, device)
    
    vlm = VisionLanguageMachine(vision_classifier, decoder)
    vlm_optimizer = torch.optim.Adam(vlm.parameters(), lr=0.0001)
    train_vlm(vlm, llava_loader, vlm_optimizer, 10, device)
    _, logits = test_vlm(vlm, test_llava_loader, device)
    
    np.save('20200170.npy', logits)
    
    

if __name__ == "__main__":
    main()