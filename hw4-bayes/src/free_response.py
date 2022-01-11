import numpy as np

from src.data import build_dataset
from src.naive_bayes import NaiveBayes, NaiveBayesEM

def free_response():
    """
    Helper code for the Free Response Questions
    """
    # Load the dataset
    data, labels, speeches, vocab = build_dataset("data", num_docs=100, max_words=2000, vocab_size=1000)
    isfinite = np.isfinite(labels)

    print(vocab)
    ### Question 1

    # Fit and evaluate the NB model
    naive_bayes = NaiveBayes()
    naive_bayes.fit(data, labels)
    nb_likelihood = naive_bayes.likelihood(data, labels)
    nb_preds = naive_bayes.predict(data)
    nb_correct = nb_preds[isfinite] == labels[isfinite]

    # Add these numbers to table in FRQ 1
    print(f"NB log likelihood: {nb_likelihood}")
    print(f"NB accuracy: {np.mean(nb_correct)}")
    
    # Fit and evaluate the NB+EM model
    naive_bayes_em = NaiveBayesEM()
    naive_bayes_em.fit(data, labels)
    nbem_likelihood = naive_bayes_em.likelihood(data, labels)
    nbem_preds = naive_bayes_em.predict(data)
    nbem_correct = nbem_preds[isfinite] == labels[isfinite]

    # Add these numbers to table in FRQ 1
    print(f"NBEM log likelihood: {nbem_likelihood}")
    print(f"NBEM accuracy: {np.mean(nbem_correct)}")

    ### Question 2

    # Use predict_proba to see output probabilities
    nbem_probs = naive_bayes_em.predict_proba(data)[isfinite]

    # 2a 
    p_x_y = np.exp(naive_bayes_em.p_x_y)
    fx = p_x_y[:,1] - p_x_y[:,0]
    rep_index = np.argsort(fx)[-5:]
    dem_index = np.argsort(fx)[:5]

    dem = vocab[dem_index]
    rep = vocab[rep_index]

    print("Dem words:")
    print(dem)

    print("Rep Words")
    print(rep)

    print("f(x)")
    print(fx)

    # The model's "confidence" in its predicted output when right 
    right_label = labels[isfinite][nbem_correct].astype(int)
    prob_when_correct = nbem_probs[nbem_correct, right_label]

    # The model's "confidence" in its predicted output when wrong 
    nbem_incorrect = np.logical_not(nbem_correct)
    wrong_label = 1 - labels[isfinite][nbem_incorrect].astype(int)
    prob_when_incorrect = nbem_probs[nbem_incorrect, wrong_label]

    # Use these number to answer FRQ 2b
    print("When NBEM is correct:")
    print(prob_when_correct.tolist())
    print("When NBEM is incorrect:")
    print(prob_when_incorrect.tolist())


if __name__ == "__main__":
    free_response()
