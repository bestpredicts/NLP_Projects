import torch
import spacy
from torchtext.data.metrics import bleu_score

from nltk.tokenize.treebank import TreebankWordDetokenizer
import sys


def translate_sentence_ger(model, sentence, german, english, device, max_length=50):
    # Load german tokenizer
    spacy_ger = spacy.load("de")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def translate_sentence_en(model, sentence,  english, german, device, max_length=50):
    # Load german tokenizer
    spacy_en = spacy.load("en")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_en(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    print("length of english sentence is: ", len(tokens))
    print(f"The english sentence is: {sentence}")
    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, english.init_token)
    tokens.append(english.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [english.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [german.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == german.vocab.stoi["<eos>"]:
            break

    translated_sentence = [german.vocab.itos[idx] for idx in outputs]
    print(f"length of translated sentence is: ", len(translated_sentence))
    print(f"the translated german sentence is: {translated_sentence}")
    # remove start token
    return translated_sentence[1:]


def bleu(data, model, english, german, device):
    model.eval()
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence_en(model, src,  english, german, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])