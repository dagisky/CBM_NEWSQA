from os import listdir
from os.path import isfile, join
from os import path
import pandas as pd
from tqdm import tqdm
import json
import collections
import math
import random
import logging

logger = logging.getLogger(__name__)

class NewsQAExample(object):
    """Docstring for ClassName"""
    def __init__(self, arg):
        super(NewsQAExample, self).__init__()
        self.question_text = question_text
        self.doc_text = doc_text
        self.doc_tokens = doc_tokens
        self.start_position = start_position
        self.end_position = end_position
        
    def __str__(self):
        return self.__repr__()

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_bad_question=False,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_bad_question = is_bad_question
        self.is_impossible = is_impossible
        

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def Union(lst1, lst2):
    """ Returns the Union of two lists containing unhashable elements
    Input:
        lst1 (list), lst2 (list)
    Output:
        lst1 (list): union of the two lists
    """
    for ele in lst2:
        if ele not in lst1:
            lst1.append(ele)
    return lst1

def get_doc_tokens(paragraph_text):
    """Tokenize the given paragraph and return character to word token offset for answer ranges"""
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    return doc_tokens, char_to_word_offset


def get_story(index, path="data/"):
    """Load CNN story
    Input: 
        index (str): Id of the story (which is also the file name in the dataset)
        path (str): path to the CNN story dataset
    Output:
        story (str): CNN story for the given index
    """
    f = open(path[1:]+index, 'r', encoding="utf-8", errors="surrogateescape")
    story = ""
    for line in f.readlines():
        story += line+" "
    f.close()
    return story

def get_char_answer_range(ans_ranges):
    char_answer= []
    for crowdsourcer_ans in ans_ranges.split("|"):
        for ans in crowdsourcer_ans.split(","):
            ans = list(ans.split(":"))
            if ans != ['None']:
                for i in range(len(ans)):
                    ans[i] = int(ans[i])
                char_answer.append(ans)
            else:
                char_answer.append(None)
    return char_answer
            
def load_data(story_path = "data/", question_filename="newsqa-data-v1", size=None):
    """Loads the NewsQA data with the respective [CNN story](https://cs.nyu.edu/~kcho/DMQA/)
    The function makes custom tokenization that creates character to word answer index offset 
    Input: NewsQA dataset file (csv)
    Output: None 
    """
    if path.exists(question_filename+".pkl"):
        df = pd.read_pickle(question_filename+".pkl")
    else:
        df = pd.read_csv(question_filename+".csv")
        if size != None:
            df = df.head(size)
        df = df.dropna(subset=['question'], axis=0)
        df.answer_char_ranges = df.answer_char_ranges.apply(lambda x: get_char_answer_range(x))
        df.validated_answers = df.validated_answers.apply(lambda x: dict(zip(json.loads(x).values(), json.loads(x).keys())) if str(x) != 'nan' else None)
        df['validater_one'] = df.validated_answers.apply(lambda x: x[1] if x != None and 1 in x.keys() else None)
        df['validater_two'] = df.validated_answers.apply(lambda x: x[2] if x != None and 2 in x.keys() else None)
        df['validater_one'] = df.validater_one.apply(lambda x : None if x == None else 'bad_question' if x == 'bad_question' else x.split(":"))
        df['validater_two'] = df.validater_two.apply(lambda x : None if x == None else 'bad_question' if x == 'bad_question' else x.split(":"))
        tokens = []
        ans_word_ranges = []
        validated_ans_word_ranges = []
        all_answers = []
        for i, story_idx in tqdm(enumerate(df.story_id)):
            story = get_story(story_idx, story_path)
            token, char_word_offset = get_doc_tokens(story)
            tokens.append(token)
            ans_word_ranges.append([[char_word_offset[x[0]] , char_word_offset[x[1]]] if x != None else None for x in  df.iloc[i].answer_char_ranges]) 
            v1, v2 = df.iloc[i].validater_one, df.iloc[i].validater_two
            validations = list()
            validations.append('bad_question' if v1 == 'bad_question' else [char_word_offset[int(v1[0])] , char_word_offset[int(v1[1])]] if  v1 != None and v1 != ['none'] else None)
            validations.append('bad_question' if v2 == 'bad_question' else [char_word_offset[int(v2[0])] , char_word_offset[int(v2[1])]] if  v2 != None and v2 != ['none'] else None)
            validated_ans_word_ranges.append(validations)
            answers = Union(validated_ans_word_ranges[-1], ans_word_ranges[-1])            
            all_answers.append(answers)
        df['ans_word_ranges'] = ans_word_ranges
        df['validated_word_answer'] = validated_ans_word_ranges
        df['all_answers'] = all_answers
        df['tokens'] = tokens
        df = df.drop(['answer_char_ranges','validated_answers','validater_one','validater_two'], axis=1)
        df.to_pickle(question_filename+".pkl")
    return df 


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    # cnt_pos, cnt_neg = 0, 0
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)

    features = []
    for example_index, example in examples.iterrows():

        # if example_index % 100 == 0:
        #     logger.info('Converting %s/%s pos %s neg %s', example_index, len(examples), cnt_pos, cnt_neg)

        query_tokens = tokenizer.tokenize(example.question)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        
        if [ans for ans in example.all_answers if ans] != []:
            answer = random.choice([ans for ans in example.all_answers if ans]) # select random answer from all the answers 
        else:
            answer = random.choice(example.all_answers)
            
        if is_training and answer == None:
            tok_start_position = -1
            tok_end_position = -1
            
        is_bad_question = False   
        if is_training and answer == 'bad_question':
            tok_start_position = -1
            tok_end_position = -1
            is_bad_question = True
        
        if is_training and not answer == None and not answer == 'bad_question':
            tok_start_position = orig_to_tok_index[answer[0]]
            if answer[1] < len(example.tokens) - 1:
                tok_end_position = orig_to_tok_index[answer[1] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            
        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

            # Query
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            span_is_impossible = True if answer == None else False
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index

            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and span_is_impossible:
                    logger.info("impossible example")
                if is_training and not span_is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info("answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    is_bad_question=is_bad_question,
                    is_impossible=span_is_impossible))
            unique_id += 1

    return features