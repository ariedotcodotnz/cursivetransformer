# Sam Greydanus | 2024

########## IMPORTS AND A FEW GLOBAL VARIABLES ##########

import os, sys, json, pickle, zipfile, functools, copy, random
import numpy as np
from math import comb

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


########## LOADING DATA AND COMBINING WORDS ##########

@functools.lru_cache(maxsize=5)
def load_and_parse_data(dataset_name):
    file_path = f'{CURRENT_DIR}/data/{dataset_name}.json.zip'
    print(f'Trying to load dataset file from {file_path}')

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        json_filename = zip_ref.namelist()[0]
        with zip_ref.open(json_filename) as file:
            data = json.load(file)

    for item in data:
        strokes = np.array(item['points'])
        strokes[:, 0] *= item['metadata']['aspectRatio']
        strokes[:, 0] -= strokes[0, 0]
        strokes[:, 1] -= 0.65
        item['points'] = strokes
    print(f'Succeeded in loading the {dataset_name} dataset; contains {len(data)} items.')
    return data
    
def combine_handwriting_examples(examples):
    return {
        'metadata': {
            'author': examples[0]['metadata']['author'],
            'asciiSequence': ' '.join(ex['metadata']['asciiSequence'] for ex in examples),
            'pointCount': sum(ex['metadata']['pointCount'] for ex in examples),
            'strokeCount': sum(ex['metadata']['strokeCount'] for ex in examples),
            'aspectRatio': examples[0]['metadata']['aspectRatio']
        },
        'points': [ex['points'].copy() for ex in examples]
    }

def generate_word_combos(raw_json, desired_num_combos=10000, num_words=3):
    num_combos = comb(len(raw_json), num_words)
    print(f'For a dataset of {len(raw_json)} examples we can generate {num_combos} combinations of {num_words} examples.')
    print(f'Generating {desired_num_combos} random combinations.')
    combo_json = []
    for _ in range(desired_num_combos):
        ixs = np.random.choice(len(raw_json), size=num_words, replace=False)
        examples_to_merge = [raw_json[ix] for ix in ixs]
        combo_json.append(combine_handwriting_examples(examples_to_merge))
    return combo_json


def sample_combo_indices(num_items, desired_num_combos, num_words):
    """Vectorized sampling of (desired_num_combos, num_words) index rows with no
    repeated item within a row. Replaces 497k serial np.random.choice(replace=False)
    calls (each one permutes all num_items entries) with one vectorized draw plus
    rejection-resampling of the rare rows that contain a duplicate."""
    assert num_items > num_words, 'need more base words than words per example'
    ixs = np.random.randint(0, num_items, size=(desired_num_combos, num_words))
    while True:
        srt = np.sort(ixs, axis=1)
        bad = (np.diff(srt, axis=1) == 0).any(axis=1)
        if not bad.any():
            return ixs
        ixs[bad] = np.random.randint(0, num_items, size=(int(bad.sum()), num_words))


########## TOKENIZATION, AUGMENTATION, AND DATA IO ##########


def decompose_offsets(offsets):
    dx, dy = offsets[:, 0], offsets[:, 1]
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    return np.column_stack((r, theta, offsets[:, 2]))

def reconstruct_offsets(polar_data):
    r, theta = polar_data[:, 0], polar_data[:, 1]
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return np.column_stack((dx, dy, polar_data[:, 2]))

def strokes_to_offsets(points, prev_points=None):
    offsets = np.zeros_like(points)
    offsets[1:, 0:2] = np.diff(points[:, 0:2], axis=0)  # Same dx, dy computation
    
    if prev_points is not None:
        offsets[0, 1] = points[0, 1] - prev_points[-1, 1]
        offsets[0, 0] = (prev_points[:, 0].max() - prev_points[-1, 0]) + \
                        (points[0, 0] - points[:, 0].min())
    
    offsets[:, 2] = points[:, 2]
    return decompose_offsets(offsets)

def offsets_to_strokes(offsets_dec):
    # Calculate cumulative sums over (dx, dt) to get absolute pen positions
    offsets = reconstruct_offsets(offsets_dec)

    absolute_coords = np.cumsum(offsets[:, :2], axis=0)  # just over (dx, dy) dimensions
    stroke_data = np.hstack((absolute_coords, offsets[:, 2:3]))
    return stroke_data

def random_horizontal_shear(stroke, shear_range=(-0.4, 0.4)):
    shear_factor = np.random.uniform(*shear_range)
    shear_matrix = np.array([[1, shear_factor], [0, 1]])
    stroke[:, :2] = np.dot(stroke[:, :2], shear_matrix.T)
    return stroke


########## STYLE-CONSISTENT AUGMENTATION (few-shot style adaptation) ##########
# When style conditioning is on (args.style_words > 0) we draw ONE set of style
# parameters per example and apply it to BOTH the style-reference words and the
# target words. The style encoder can then learn to read slant / proportions /
# stroke density off the reference strokes, which is exactly the skill needed to
# mimic a new writer's sample at inference time. The ranges are deliberately wider
# than the legacy augmentation so the style code carries real information.

def sample_style_params(args):
    return {
        'shear': np.random.uniform(-0.45, 0.25),
        'scale_x': np.random.uniform(0.85, 1.2),
        'scale_y': np.random.uniform(0.85, 1.2),
        'downsample_frac': args.downsample_mean + args.downsample_width * (np.random.rand() - .5),
    }


def apply_style_params(stroke, params):
    shear_matrix = np.array([[1, params['shear']], [0, 1]])
    stroke[:, :2] = np.dot(stroke[:, :2], shear_matrix.T)
    stroke[:, 0:1] *= params['scale_x']
    stroke[:, 1:2] *= params['scale_y']
    return downsample(stroke, params['downsample_frac'])

def random_rotate(stroke, angle_range=(-.08, .08)):
    angle = np.random.uniform(*angle_range)
    rad = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)]])
    stroke[:, :2] = np.dot(stroke[:, :2], rotation_matrix.T)
    return stroke


def downsample(arr, fraction, drop_prob=0.05):
    if fraction == 1:
        return arr
    result, stroke = [], []
    for point in arr:
        if point[2] == 1:
            stroke.append(point)
        else:
            if stroke:
                new_len = max(2, int(len(stroke) * (1 - fraction)))
                indices = np.linspace(0, len(stroke) - 1, new_len, dtype=int)
                reduced_stroke = np.array(stroke)[indices]
                if drop_prob > 0:
                    reduced_stroke = [p for i, p in enumerate(reduced_stroke) if i == 0 or i == len(reduced_stroke) - 1 or random.random() > drop_prob]
                result.extend(reduced_stroke)
            result.append(point)
            stroke = []
    if stroke:
        new_len = max(2, int(len(stroke) * (1 - fraction)))
        indices = np.linspace(0, len(stroke) - 1, new_len, dtype=int)
        reduced_stroke = np.array(stroke)[indices]
        if drop_prob > 0:
            reduced_stroke = [p for i, p in enumerate(reduced_stroke) if i == 0 or i == len(reduced_stroke) - 1 or random.random() > drop_prob]
        result.extend(reduced_stroke)
    return np.array(result)


class StrokeDataset(Dataset):
    def __init__(self, raw_word_strokes, texts, args, max_text_length=50, name='',
                 style_word_strokes=None):
        self.raw_word_strokes = raw_word_strokes  # List of lists of Nx3 arrays, each inner list representing words in a sentence
        self.texts = texts      # List of corresponding text strings
        self.args = args
        self.alphabet = args.alphabet  # String of all possible characters
        self.augment = args.augment
        self.max_seq_length = args.max_seq_length
        self.max_text_length = max_text_length
        self.name = name
        self.counter = 0

        # Style reference words (few-shot style adaptation). Parallel to
        # raw_word_strokes: for each example, a list of OTHER words by the same writer
        # that will receive the SAME style params as the target words.
        self.style_word_strokes = style_word_strokes
        self.max_style_length = getattr(args, 'max_style_length', 0) if getattr(args, 'style_words', 0) > 0 else 0

        self.theta_bins = np.linspace(-np.pi, np.pi, 220)

        r_bins_pen_down = np.concatenate([
                            np.asarray([0]),
                            np.linspace(0.0001, 0.060, 30),
                            np.geomspace(0.06001, 0.90, 120) ]) # 100 discrete radii
        r_bins_pen_up = r_bins_pen_down + max(r_bins_pen_down) + 1  # Offset for pen-up states
        self.r_bins = np.concatenate([r_bins_pen_down, r_bins_pen_up])  # 200 bins for: {radii x pen up/down}

        self.feature_sizes = [len(self.r_bins), len(self.theta_bins)]
        self.cumulative_sizes = np.cumsum([0] + self.feature_sizes)

        # Add special tokens for strokes
        self.PAD_TOKEN = sum(self.feature_sizes)
        self.END_TOKEN = sum(self.feature_sizes) + 1
        self.WORD_TOKEN = sum(self.feature_sizes) + 2

        # Character tokenization
        self.char_PAD_TOKEN = 0
        self.stoi = {ch:i+1 for i,ch in enumerate(self.alphabet)}
        self.itos = {i:s for s,i in self.stoi.items()}

    def split_by_word_tokens(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        # Find pairs of WORD_TOKENs
        word_boundaries = np.where((tokens[:-1] == self.WORD_TOKEN) & (tokens[1:] == self.WORD_TOKEN))[0]
        # Split using these boundaries
        splits = np.split(tokens, word_boundaries + 1)
        return [s for s in splits if len(s) > 0]

    def concat_with_word_tokens(self, token_lists):
        word_tokens = np.array([self.WORD_TOKEN, self.WORD_TOKEN])
        return np.concatenate([np.concatenate([tokens, word_tokens]) if i < len(token_lists)-1 else tokens 
                             for i, tokens in enumerate(token_lists)])

    def augment_stroke(self, stroke, style_params=None):
        if style_params is not None:  # style-consistent path: shared params per example
            return apply_style_params(stroke, style_params)
        # legacy per-word augmentation (style conditioning off)
        # stroke = random_horizontal_shear(stroke, shear_range=(-0.30, 0.15)) # Horizontal shear
        stroke = random_horizontal_shear(stroke, shear_range=(-0.22, -0.18))
        stroke[:, 0:1] *= np.random.uniform(0.9, 1.1)
        stroke[:, 1:2] *= np.random.uniform(0.9, 1.1)
        # stroke = random_rotate(stroke, angle_range=(-.08, .08))

        downsample_percent = self.args.downsample_mean + self.args.downsample_width * (np.random.rand()-.5)
        stroke = downsample(stroke, downsample_percent)
        return stroke

    def __len__(self):
        return len(self.raw_word_strokes)

    def get_vocab_size(self):
        return sum(self.feature_sizes) + 3  # +3 for PAD, END, and WORD tokens

    def get_char_vocab_size(self):
        return len(self.alphabet) + 1  # +1 for PAD token

    def get_stroke_seq_length(self):
        return self.max_seq_length

    def get_text_seq_length(self):
        return self.max_text_length

    def get_style_seq_length(self):
        return self.max_style_length

    def encode_style_words(self, word_strokes):
        """Tokenize a list of word stroke arrays into a fixed-length style-reference
        sequence (same vocab as the main stroke stream, PAD-padded / truncated)."""
        encoded = [self.encode_stroke(
                   strokes_to_offsets(word_strokes[i],
                   prev_points=word_strokes[i-1] if i > 0 else None))
                        for i in range(len(word_strokes))]
        tokens = self.concat_with_word_tokens(encoded) if encoded else np.zeros(0, dtype=np.int64)
        s = torch.full((self.max_style_length,), self.PAD_TOKEN, dtype=torch.long)
        n = min(len(tokens), self.max_style_length)
        s[:n] = torch.tensor(tokens[:n], dtype=torch.long)
        return s

    def encode_stroke(self, stroke):
        # Encode magnitude and pen state together
        r_idx = np.digitize(stroke[:, 0], self.r_bins[:len(self.r_bins)//2]) - 1
        r_idx[stroke[:, 2] == 0] += len(self.r_bins) // 2  # Offset for pen-up states

        theta_idx = np.digitize(stroke[:, 1], self.theta_bins) - 1

        encoded = np.column_stack([
            theta_idx + self.cumulative_sizes[1],
            r_idx + self.cumulative_sizes[0],])
        return encoded.flatten()

    def decode_stroke(self, ix):
      if isinstance(ix, torch.Tensor):
          ix = ix.cpu().numpy()
      # The model is only trained up to its END token; anything after it is noise.
      end_pos = np.where(ix == self.END_TOKEN)[0]
      if len(end_pos) > 0:
          ix = ix[:end_pos[0]]
      ix_list = self.split_by_word_tokens(ix)
      words = [self.decode_word_strokes(w) for w in ix_list]
      # A stray WORD token can create an empty group that would shift every later word
      # by one position; an empty "word" is never legitimate, so drop them.
      return [w for w in words if len(w) > 0]

    def decode_word_strokes(self, ix):
        if isinstance(ix, torch.Tensor):
            ix = ix.cpu().numpy()

        # Remove PAD, END, and WORD tokens
        ix = ix[(ix != self.PAD_TOKEN) & (ix != self.END_TOKEN) & (ix != self.WORD_TOKEN)]

        # Reshape the flattened array back to Nx2
        ix = ix[:(len(ix)//2)*2]
        ix = ix.reshape(-1, 2)

        r_idx = ix[:, 1] - self.cumulative_sizes[0]
        pen = (r_idx < len(self.r_bins) // 2).astype(int)
        r_idx[pen == 0] -= len(self.r_bins) // 2
        r = self.r_bins[:len(self.r_bins)//2][r_idx.clip(0, len(self.r_bins)//2 - 1)]
        theta = self.theta_bins[(ix[:, 0] - self.cumulative_sizes[1]).clip(0, len(self.theta_bins)-1)]

        return np.column_stack([r, theta, pen])

    def encode_text(self, text, do_padding=True):
        encoded_text = torch.tensor([self.stoi.get(ch, self.char_PAD_TOKEN) for ch in text], dtype=torch.long)
        if do_padding:
            c = torch.full((self.max_text_length,), self.char_PAD_TOKEN, dtype=torch.long)
            text_len = min(len(encoded_text), self.max_text_length)
            c[:text_len] = encoded_text[:text_len]
        else:
            c = encoded_text
        return c

    def decode_text(self, ix):
        if isinstance(ix, torch.Tensor):
            ix = ix.cpu().numpy()
        
        first_pad = np.where(ix == self.char_PAD_TOKEN)[0]
        end_idx = first_pad[0] if len(first_pad) > 0 else len(ix)
        return ''.join(self.itos.get(i, '') for i in ix[:end_idx])

    def __getitem__(self, idx):
        word_strokes = self.raw_word_strokes[idx]
        text = self.texts[idx]
        use_style = self.max_style_length > 0 and self.style_word_strokes is not None
        style_words = self.style_word_strokes[idx] if use_style else []

        # Apply augmentation per word if enabled
        if self.augment:
            np.random.seed(self.args.seed+idx+self.counter)  # use the same augmentation across all words in sample
            # With style conditioning on, ONE set of style params is shared by the
            # reference words and the target words, so the reference is informative.
            style_params = sample_style_params(self.args) if use_style else None
            word_strokes = [self.augment_stroke(word.copy(), style_params) for word in word_strokes]
            style_words = [self.augment_stroke(word.copy(), style_params) for word in style_words]
        self.counter = (self.counter + 1) % 100000

        # Encode each word separately and combine with WORD_TOKENs
        encoded_words = [self.encode_stroke(
                         strokes_to_offsets(word_strokes[i],
                         prev_points=word_strokes[i-1] if i > 0 else None))
                              for i in range(len(word_strokes)) ]
        encoded_stroke = self.concat_with_word_tokens(encoded_words)

        # Create input and target sequences. Inputs are padded with PAD_TOKEN; targets
        # are padded with -1 so the loss IGNORES the padded tail (otherwise 30-50% of
        # every loss is "predict PAD", which badly dilutes the gradient signal).
        x = torch.full((self.max_seq_length,), self.PAD_TOKEN, dtype=torch.long)
        y = torch.full((self.max_seq_length,), -1, dtype=torch.long)

        seq_len = min(len(encoded_stroke), self.max_seq_length - 1)  # -1 to leave room for END token
        x[:seq_len] = torch.tensor(encoded_stroke[:seq_len], dtype=torch.long)
        x[seq_len] = self.END_TOKEN

        y[:seq_len] = x[1:seq_len+1]  # the last real position's target is the END token
        # Teach the model to go quiet after END (END->PAD, then PAD->PAD) but only for a
        # few positions, so these easy PAD predictions don't flood the loss the way the
        # old full-PAD targets did. The rest of the tail stays -1 (ignored).
        tail_end = min(self.max_seq_length, seq_len + 1 + 8)
        y[seq_len:tail_end] = self.PAD_TOKEN

        c = self.encode_text(text)
        # Style tensor comes LAST so legacy (x, c, y) index-based unpacking still works;
        # it is zero-length when style conditioning is off.
        s = self.encode_style_words(style_words) if use_style \
            else torch.zeros(0, dtype=torch.long)
        return x, c, y, s


def create_datasets(args):
  np.random.seed(args.seed) ; torch.manual_seed(args.seed)
  data = load_and_parse_data(args.dataset_name)

  # partition the input data into a training and the test set
  test_set_size = min(1000, max(10, int(len(data) * 0.05))) # between 10 and 1000 examples: ideally 5% of dataset
  rp = torch.randperm(len(data)).tolist()
  train_words = [data[i] for i in rp[:-test_set_size]]
  test_words = [data[i] for i in rp[-test_set_size:]]

  n_style_words = getattr(args, 'style_words', 0)

  def build_split(words, num_combos):
      # Combos hold REFERENCES to the base word arrays rather than copies; __getitem__
      # copies a word only when augmenting it. This turns dataset construction from
      # minutes + tens of GB (497k examples x num_words deep copies) into seconds.
      print(f'For a dataset of {len(words)} examples we can generate {comb(len(words), args.num_words)} combinations of {args.num_words} examples.')
      print(f'Generating {num_combos} random combinations.')
      # Draw targets and references together so each reference is a distinct
      # "other" word and cannot leak a target example into the style input.
      row_width = args.num_words + n_style_words
      all_ixs = sample_combo_indices(len(words), num_combos, row_width)
      ixs = all_ixs[:, :args.num_words]
      word_strokes = [[words[j]['points'] for j in row] for row in ixs]
      texts = [' '.join(words[j]['metadata']['asciiSequence'] for j in row) for row in ixs]

      style_strokes = None
      if n_style_words > 0:  # a few OTHER words by the same writer, as style reference
          style_ixs = all_ixs[:, args.num_words:]
          style_strokes = [[words[j]['points'] for j in row] for row in style_ixs]
      return word_strokes, texts, style_strokes

  train_word_strokes, train_texts, train_style = build_split(train_words, args.train_size)
  test_word_strokes, test_texts, test_style = build_split(test_words, args.test_size)

  print(f"Number of examples in the train dataset: {len(train_word_strokes)}")
  print(f"Number of examples in the test dataset: {len(test_word_strokes)}")
  print(f"Average number of words per example: {np.mean([len(strokes) for strokes in train_word_strokes]):.1f}")
  print(f"Max token sequence length: {args.max_seq_length}")
  print(f"Number of unique characters in the ascii vocabulary: {len(args.alphabet)}")
  print("Ascii vocabulary:")
  print(f'\t"{args.alphabet}"')
  print(f"Split up the dataset into {len(train_word_strokes)} training examples and {len(test_word_strokes)} test examples")

  # wrap in dataset objects
  train_dataset = StrokeDataset(train_word_strokes, train_texts, args, name='train', style_word_strokes=train_style)
  test_dataset = StrokeDataset(test_word_strokes, test_texts, args, name='test', style_word_strokes=test_style)
  return train_dataset, test_dataset


########## LOADING A USER'S HANDWRITING AS A STYLE REFERENCE ##########

def normalize_raw_examples(data):
    """Apply the same normalization as load_and_parse_data to freshly captured
    handwriting (e.g. the output of data/collect.html)."""
    for item in data:
        strokes = np.array(item['points'], dtype=np.float64)
        strokes[:, 0] *= item['metadata']['aspectRatio']
        strokes[:, 0] -= strokes[0, 0]
        strokes[:, 1] -= 0.65
        item['points'] = strokes
    return data


def load_style_reference(path, dataset):
    """Load a few words/sentences of someone's handwriting and tokenize them into a
    style-reference tensor of shape (1, max_style_length) for style-conditioned
    generation. `path` is a .json or .json.zip file in the same format as the
    training data (a list of {'points': Nx3, 'metadata': {...}} examples)."""
    if path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            with zip_ref.open(zip_ref.namelist()[0]) as file:
                data = json.load(file)
    else:
        with open(path) as file:
            data = json.load(file)
    data = normalize_raw_examples(data)
    word_arrays = [item['points'] for item in data]
    assert dataset.max_style_length > 0, \
        'This model was trained without style conditioning (style_words=0)'
    return dataset.encode_style_words(word_arrays).unsqueeze(0)


class InfiniteDataLoader:
    """
    From Andrej Karpathy: this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:  # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch
