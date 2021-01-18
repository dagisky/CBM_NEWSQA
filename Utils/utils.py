import sys
import torch
import numpy as np
import time
import tensorflow as tf
from io import BytesIO
import scipy.misc


class TimeMeter:    

    def __init__(self):
        """Counts time duaration"""
        self.start_time, self.duration, self.counter=0. ,0. ,0.

    def start(self):
        """Start timer"""
        self.start_time=time.perf_counter()

    def stop(self):
        """Stop timer"""
        self.duration=time.perf_counter()-self.start_time
        self.counter+=1

    def get(self):
        """Returns time duration"""
        return self.duration/self.counter

    def reset(self):
        """Reset timer"""
        self.start_time, self.duration, self.counter=0. ,0. ,0.        

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        tag=self.tag(tag)
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        tag=self.tag(tag)
        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.compat.v1.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.compat.v1.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        tag=self.tag(tag)
        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        # self.writer.flush()
    def flush(self):
        """Flushes the event file to disk."""
        self.writer.flush()
        
    def tag(self,tag):
        return tag.replace('.','/')

    def model_graph(self, model, input_list):
        self.writer.add_graph(model, input_list)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


class MLP(torch.nn.Module):
    NLS = {'relu': torch.nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax, 'logsoftmax': nn.LogSoftmax}

    def __init__(self, D_in: int, hidden_dims: list, D_out: int, nonlin='relu'):
        super().__init__()
        
        all_dims = [D_in, *hidden_dims, D_out]
        layers = []
        
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            layers += [
                nn.Linear(in_dim, out_dim, bias=True),
                MLP.NLS[nonlin]()
            ]
        
        self.fc_layers = nn.Sequential(*layers[:-1])
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        z = self.fc_layers(x)
        y_pred = self.log_softmax(z)
        # Output is always log-probability
        return y_pred

# class MLP(nn.Module):
#     def __init__(self, equation, in_features, hidden_size, out_size):
#         super(MLP, self).__init__()
#         self.equation = equation
#         # Layers
#         # 1
#         self.W1 = nn.Parameter(torch.zeros(in_features, hidden_size))
#         nn.init.xavier_uniform_(self.W1.data)
#         self.b1 = nn.Parameter(torch.zeros(hidden_size))
#         # 2
#         self.W2 = nn.Parameter(torch.zeros(hidden_size, out_size))
#         nn.init.xavier_uniform_(self.W2.data)
#         self.b2 = nn.Parameter(torch.zeros(out_size))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         hidden = torch.tanh(torch.einsum(self.equation, x, self.W1) + self.b1)
#         out = torch.tanh(torch.einsum(self.equation, hidden, self.W2) + self.b2)
#         return out


class OptionalLayer(nn.Module):
    def __init__(self, layer: nn.Module, active: bool = False):
        super(OptionalLayer, self).__init__()
        self.layer = layer
        self.active = active
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active:
            return self.layer(x)
        return x



class LayerNorm1(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super(LayerNorm1, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = (x - mu).pow(2).mean(-1, keepdim=True)
        normalized = (x - mu) / (torch.sqrt(sigma + self.eps))
        return normalized * self.gain + self.bias

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: optim.Optimizer, multiplier: float, steps: int):
        self.multiplier = multiplier
        self.steps = steps
        super(WarmupScheduler, self).__init__(optimizer=optimizer)

    def get_lr(self):
        if self.last_epoch < self.steps:
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return self.base_lrs

    def decay_lr(self, decay_factor: float):
        self.base_lrs = [decay_factor * base_lr for base_lr in self.base_lrs]