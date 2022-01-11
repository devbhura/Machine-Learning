import os
import re
import numpy as np
import scipy.sparse as sparse

from collections import defaultdict


def build_dataset(datadir, num_docs=0, max_words=None, vocab_size=None):
    """
    Build a document-word matrix from State of the Union speeches, taken from
    https://en.m.wikisource.org/wiki/Portal:State_of_the_Union_Speeches_by_United_States_Presidents

    You should not need to modify this function!

    The dataset is returned as `scipy.sparse.csr_matrix`, which is a more efficient
    way of representing a feature matrix when most entries are zero.
    It should be much faster to work with the sparse matrix for large datasets,
    but you can convert the matrix to a numpy array with `data.toarray()`.
    In the data matrix returned by this function, `data[i, j]` contains the number
    of times that word j appears in document i. For more details on sparse matrices,
    see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html

    Args:
        datadir: directory where speeches and stopwords are located
        num_docs: number of documents to include. The most recent speeches are included.
        max_words: number of words per document to keep after processing
        vocab_size: the total number of words to consider in the vocabulary
            (This function may return a slightly larger vocabulary. if vocab_size
             is 10, the vocabulary will include the 10 most commonly-occurring words.
             If there is a tie in number of occurrences, all words tied
             for 10th-most-common will be included.)

    Returns:
        data: A compressed sparse row matrix of floats with shape
              `[num_documents, vocabulary_size]`
        labels: An array of float labels with shape
              `[num_documents, ]`
        speeches: A list of `num_documents` speech filenames (e.g. 1790_Washington)
        vocab: The words in the vocabulary; a list of words of length
              `vocabulary_size`
    """
    speeches_dir = os.path.join(datadir, "speeches")
    if not os.path.exists(speeches_dir):
        if os.path.isfile(f"{speeches_dir}.zip"):
            os.system(f"unzip -q {speeches_dir}.zip -d {speeches_dir}/")

    files = sorted([f for f in os.listdir(speeches_dir) if os.path.isfile(os.path.join(speeches_dir, f))])
    if num_docs == 0:
        num_docs = len(files)

    with open(os.path.join(datadir, "stopwords.txt")) as inf:
        stopwords = inf.readline().split()

    speeches = []
    labels = []
    docs = defaultdict(list)

    # Load the raw text from each speech
    for i, fn in enumerate(files[len(files)-num_docs:], start=1):
        year = int(fn[:4])
        president = fn[5:]
        speeches.append(fn)
        label = label_from_party(party_from_president(president, year))
        labels.append(label)
        with open(os.path.join(speeches_dir, fn), encoding="utf-8") as f:
            text = ' '.join(f.readlines())
            docs['doc' + str(i)] = clean_text(text, max_words, stopwords)

    # count up the occurrences of each word in all documents
    vocab = defaultdict(int)
    for docterms in docs.values():
        for term in docterms:
            vocab[term] += 1

    # build a vocabulary of at most vocab_size words
    if vocab_size is not None and vocab_size < len(vocab): 
        cutoff = sorted(list(vocab.values()))[-vocab_size]
        vocab = set([word for word, count in vocab.items() if count >= cutoff])

    # only keep the words that are in our vocabulary
    n_nonzero = 0
    for docterms in docs.values():
        for term in docterms:
            unique = set([term for term in docterms if term in vocab])
            n_nonzero += len(unique)

    docnames = np.array(list(docs.keys()))
    vocab = np.array(list(vocab))  

    vocab_sorter = np.argsort(vocab)
    ndocs, nvocab = len(docnames), len(vocab)

    # The sparse matrix is determined by three arrays.
    #   `data` contains the actual counts of words
    #   `rows` contains the row (document index) pointers
    #   `cols` contains the column (word index) pointers
    data = np.zeros(n_nonzero, dtype=np.int32)
    rows = np.zeros(n_nonzero, dtype=np.int32)
    cols = np.zeros(n_nonzero, dtype=np.int32)

    ind = 0
    for docname, terms in docs.items():
        # find the word indices of the words in our vocab
        terms = [term for term in terms if term in vocab]
        term_indices = vocab_sorter[np.searchsorted(vocab, terms, sorter=vocab_sorter)]

        # Count the occurences of each unique word
        uniq_indices, counts = np.unique(term_indices, return_counts=True)
        n_vals = len(uniq_indices)
        ind_end = ind + n_vals

        # Update our data and pointer arrays
        data[ind:ind_end] = counts
        cols[ind:ind_end] = uniq_indices
        doc_idx = np.where(docnames == docname)
        rows[ind:ind_end] = np.repeat(doc_idx, n_vals)

        ind = ind_end

    # Build the sparse matrix from data and pointer arrays
    data = sparse.csr_matrix((data, (rows, cols)), shape=(ndocs, nvocab), dtype=np.intc)
    return (data, np.array(labels), speeches, vocab)


def train_test_unlabeled_split(data, labels, speeches, splits):
    """
    Split the data and labels into train and test splits using the
        provided splits dictionary.
        You should not need to modify this function!

    Args:
        data: a (sparse) matrix of `[num_documents, vocab_size]`
              containing word counts
        labels: an array of labels corresponding to documents
        speeches: a list of speech names, e.g. "1790_Washington"
        splits: a dictionary mapping speeches to splits, where
                0 means labeled training data, 1 means test data,
                2 means unlabeled training data, and either -1
                or omitting the speech name from this dict
                means that speech will be omitted from the data.

    Returns:
        train_data: a sparse matrix of the words in the training documents
        train_labels: the labels of the training documents
                      np.nan is used to denote "unlabeled" data
        train_data: a sparse matrix of the words in the test documents
        train_labels: the labels of the test documents. All test labels
                      should be finite.
    """
    split_vector = -1 * np.ones_like(labels)
    for idx, speech in enumerate(speeches):
        split_vector[idx] = splits.get(speech, -1)

    train_data = data[split_vector == 0]
    train_labels = labels[split_vector == 0]
    test_data = data[split_vector == 1]
    test_labels = labels[split_vector == 1]
    unlabeled_data = data[split_vector == 2]
    unlabeled_labels = np.nan * np.ones_like(labels[split_vector == 2])

    return (sparse.vstack([train_data, unlabeled_data]),
            np.concatenate([train_labels, unlabeled_labels], axis=0),
            test_data, test_labels)


def party_from_president(name, year):
    """
    Assigning political party label from president lastname. Taken from:
    https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States
    Using "?" to represent parties before 1964.
    """

    if year < 1964:
        return "?"

    d = {
        "Trump": "R",
        "Obama": "D",
        "Bush": "R",
        "Clinton": "D",
        "Reagan": "R",
        "Carter": "D",
        "Ford": "R",
        "Nixon": "R",
        "Johnson": "D",
        }

    return d[name]


def label_from_party(party):
    """
    Convert party label from string into a float
    Pre-1964 "?" party labels are represented with np.nan
    """
    if party == "?":
        return np.nan
    elif party == "D":
        return 0.
    elif party == "R":
        return 1.
    else:
        raise ValueError()


def clean_text(text, max_words=None, stopwords=None):
    """
    Remove stopwords, punctuation, and numbers from text.

    Args:
        text: article text
        max_words: number of words to keep after processing
                   if None, include all words
        stopwords: a list of words to skip during processing
                   if None, ignored

    Returns:
        Space-delimited and cleaned string
    """
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = re.split(r"\s+", text)

    good_tokens = []
    for token in tokens:
        token = token.lower().strip()

        # remove stopwords
        if stopwords is not None and token in stopwords:
            continue

        # remove tokens without alphabetic characters (i.e. punctuation, numbers)
        if any(char.isalpha() for char in token):
            good_tokens.append(token)
            
    # skipping first ~20 words, which are often introductory
    if max_words is None:
        return good_tokens[20:]
    else:
        return good_tokens[20:20+max_words]
