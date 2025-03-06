import torch
import numpy as np
from clip_config import axis_combinations
from params import tokenizer, text_encoder, torch_device


def get_word_embeddings(
    prompt, tokenizer=tokenizer, text_encoder=text_encoder, torch_device=torch_device
):
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(torch_device)

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids)[0].reshape(1, -1)

    text_embeddings = text_embeddings.cpu().numpy()
    return text_embeddings / np.linalg.norm(text_embeddings)


def get_concat_embeddings(names, merge=False):
    embeddings = []

    for name in names:
        embedding = get_word_embeddings(name)
        embeddings.append(embedding)

    embeddings = np.vstack(embeddings)

    if merge:
        embeddings = np.average(embeddings, axis=0).reshape(1, -1)

    return embeddings


def get_axis_embeddings(A, B):
    emb = []

    for a, b in zip(A, B):
        e = get_word_embeddings(a) - get_word_embeddings(b)
        emb.append(e)

    emb = np.vstack(emb)
    ax = np.average(emb, axis=0).reshape(1, -1)

    return ax


def calculate_residual(
    axis, axis_names, from_words=None, to_words=None, residual_axis=1
):
    axis_indices = [0, 1, 2]
    axis_indices.remove(residual_axis)

    if axis_names[axis_indices[0]] in axis_combinations:
        fembeddings = get_concat_embeddings(
            axis_combinations[axis_names[axis_indices[0]]], merge=True
        )
    else:
        axis_combinations[axis_names[axis_indices[0]]] = from_words + to_words
        fembeddings = get_concat_embeddings(from_words + to_words, merge=True)

    if axis_names[axis_indices[1]] in axis_combinations:
        sembeddings = get_concat_embeddings(
            axis_combinations[axis_names[axis_indices[1]]], merge=True
        )
    else:
        axis_combinations[axis_names[axis_indices[1]]] = from_words + to_words
        sembeddings = get_concat_embeddings(from_words + to_words, merge=True)

    fprojections = fembeddings @ axis[axis_indices[0]].T
    sprojections = sembeddings @ axis[axis_indices[1]].T

    partial_residual = fembeddings - (fprojections.reshape(-1, 1) * fembeddings)
    residual = partial_residual - (sprojections.reshape(-1, 1) * sembeddings)

    return residual


def read_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content
