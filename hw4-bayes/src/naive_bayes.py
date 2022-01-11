from src.utils import softmax

import numpy as np


class NaiveBayes:
    """
    A Naive Bayes classifier for binary data.
    """

    def __init__(self, smoothing=1):
        """
        Args:
            smoothing: controls the smoothing behavior when computing p(x|y).
                If the word "jackpot" appears `k` times across all documents with
                label y=1, we will instead record `k + self.smoothing`. Then
                `p("jackpot" | y=1) = (k + self.smoothing) / Z`, where Z is a
                normalization constant that accounts for adding smoothing to
                all words.
        """
        self.smoothing = smoothing

    def predict(self, X):
        """
        Return the most probable label for each row x of X.
        You should not need to edit this function.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """
        Using self.p_y and self.p_x_y, compute the probability p(y | x) for each row x of X.
        While you will have used log probabilities internally, the returned array should be
            probabilities, not log probabilities. You may use src.utils.softmax to transform log
            probabilities to probabilities.

        Args:
            X: a data matrix of shape `[n_documents, vocab_size]` on which to predict p(y | x)

        Returns 
            probs: an array of shape `[n_documents, n_labels]` where probs[i, j] contains
                the probability `p(y=j | X[i, :])`. Thus, for a given row of this array,
                sum(probs[i, :]) == 1.
        """
        n_docs, vocab_size = X.shape
        n_labels = 2


        assert hasattr(self, "p_y") and hasattr(self, "p_x_y"), "Model not fit!"
        assert vocab_size == self.vocab_size, "Vocab size mismatch"

        probs = X@self.p_x_y + np.log(np.array([1-self.p_y[0], self.p_y[0]]))
        
        probs= softmax(probs,axis=1)
        
        return probs
        
        

    def fit(self, X, y):
        """
        Compute self.p_y and self.p_x_y using the training data.
        You should store log probabilities to avoid underflow.
        This function *should not* use unlabeled data. Wherever y is NaN, that
        label and the corresponding row of X should be ignored.

        self.p_y should contain the marginal probability of each class label.
            Because we are doing binary classification, you may choose
            to represent p_y as a single value representing p(y=1)

        self.p_x_y should contain the conditional probability of each word
            given the class label: p(x | y). This should be an array of shape
            [n_vocab, n_labels].  Remember to use `self.smoothing` to smooth word counts!
            See __init__ for details. If we see M total words across all N documents with
            label y=1, have a vocabulary size of V words, and see the word "jackpot" `k`
            times, then: `p("jackpot" | y=1) = (k + self.smoothing) / (M + self.smoothing *
            V)` Note that `p("jackpot" | y=1) + p("jackpot" | y=0)` will not sum to 1;
            instead, `sum_j p(word_j | y=1)` will sum to 1.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size
        
        # print(np.invert(np.isnan(y)))
        y_labeled = y[np.invert(np.isnan(y))]
        X_labeled = X[np.invert(np.isnan(y))]
        self.y_labeled = y_labeled
        self.X_labeled = X_labeled
        self.p_y = (y_labeled==1).sum()/len(y_labeled)
        self.p_y = np.array([self.p_y])

        # print(X_labeled.toarray())
        # print("y")
        # print(y_labeled)

        y_1 = y_labeled[y_labeled==1]
        X_1 = X_labeled[y_labeled==1]
        y_0 = y_labeled[y_labeled==0]
        X_0 = X_labeled[y_labeled==0]

        P_X_0 = (np.sum(X_0,axis=0)+self.smoothing)/(np.sum(X_0)+self.smoothing*vocab_size)
        P_X_1 = (np.sum(X_1,axis=0)+self.smoothing)/(np.sum(X_1)+self.smoothing*vocab_size)

        # print(P_X_0)
        self.p_x_y = np.vstack((P_X_0,P_X_1))

        
        self.p_x_y = np.asarray(np.transpose(self.p_x_y))
        
        self.p_x_y = np.log(self.p_x_y)


        # print(self.p_x_y)
        


    def likelihood(self, X, y):
        """
        Using fit self.p_y and self.p_x_y, compute the log likelihood of the data.
            You should use logs to avoid underflow.
            This function should not use unlabeled data. Wherever y is NaN,
            that label and the corresponding row of X should be ignored.

        Recall that the log likelihood of the data can be written:
          `sum_i (log p(y_i) + sum_j log p(x_j | y_i))`

        Note: If the word w appears `k` times in a document, the term
            `p(w | y)` should appear `k` times in the likelihood for that document!

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the (log) likelihood of the data.
        """
        assert hasattr(self, "p_y") and hasattr(self, "p_x_y"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2
        y_labeled = y[np.invert(np.isnan(y))]
        X_labeled = X[np.invert(np.isnan(y))]
        X_1 = X_labeled[y_labeled==1]
        X_0 = X_labeled[y_labeled==0]
        # print(X_labeled.toarray())
        # print(self.p_x_y)

        likelihood_y_0 = np.log(1-self.p_y[0]) + np.nansum(X_0.dot(self.p_x_y[:,0]))
        likelihood_y_1 = np.log(self.p_y[0]) + np.nansum(X_1.dot(self.p_x_y[:,1]))
        

        
        log_likelihood = likelihood_y_1 + likelihood_y_0

        # print(log_likelihood)
        return log_likelihood


class NaiveBayesEM(NaiveBayes):
    """
    A NaiveBayes classifier for binary data,
        that uses unlabeled data in the Expectation-Maximization algorithm
    """

    def __init__(self, max_iter=10, smoothing=1):
        """
        Args:
            max_iter: the maximum number of iterations in the EM algorithm,
                where each iteration contains both an E step and M step.
                You should check for convergence after each iterations,
                e.g. with `np.isclose(prev_likelihood, likelihood)`, but
                should terminate after `max_iter` iterations regardless of
                convergence.
            smoothing: controls the smoothing behavior when computing p(x|y).
                If the word "jackpot" appears `k` times across all documents with
                label y=1, we will instead record `k + self.smoothing`. Then
                `p("jackpot" | y=1) = (k + self.smoothing) / Z`, where Z is a
                normalization constant that accounts for adding smoothing to
                all words.
        """
        self.max_iter = max_iter
        self.smoothing = smoothing

    def fit(self, X, y):
        """
        Compute self.p_y and self.p_x_y using the training data.
        You should store log probabilities to avoid underflow.
        This function *should* use unlabeled data within the EM algorithm.

        During the E-step, use the superclass self.predict_proba to
            infer a distribution over the labels for the unlabeled examples.
            Note: you should *NOT* replace the true labels with your predicted
            labels. You can use a `np.where` statement to only update the
            labels where `np.isnan(y)` is True.

        During the M-step, update self.p_y and self.p_x_y, similar to the
            `fit()` call from the NaiveBayes superclass. However, when counting
            words in an unlabeled example to compute p(x | y), instead of the
            binary label y you should use p(y | x).

        For help understanding the EM algorithm, refer to the lectures and
            http://www.cs.columbia.edu/~mcollins/em.pdf
            This PDF is also uploaded to the course website under readings.
            While Figure 1 of this PDF suggests randomly initializing
            p(y) and p(x | y) before your first E-step, please initialize
            all probabilities equally; e.g. if your vocab size is 4, p(x | y=1)
            would be 1/4 for all values of x. This will make it easier to
            debug your code without random variation, and will checked
            in the `test_em_initialization` test case.

        self.p_y should contain the marginal probability of each class label.
            Because we are doing binary classification, you may choose
            to represent p_y as a single value representing p(y=1)

        self.p_x_y should contain the conditional probability of each word
            given the class label: p(x | y). This should be an array of shape
            [n_vocab, n_labels].  Remember to use `self.smoothing` to smooth word counts!
            See __init__ for details. If we see M total
            words across all documents with label y=1, have a vocabulary size
            of V words, and see the word "jackpot" `k` times, then:
            `p("jackpot" | y=1) = (k + self.smoothing) / (M + self.smoothing * V)`
            Note that `p("jackpot" | y=1) + p("jackpot" | y=0)` will not sum to 1;
            instead, `sum_j p(word_j | y=1)` will sum to 1.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size

        X_unlabeled = X[np.isnan(y)]

        self.p_y = np.array([0.5])
        self.p_x_y = (np.ones((vocab_size,n_labels)))
        self.p_x_y = np.log(self.p_x_y/vocab_size)

        iter = 0
        X_arr = X_unlabeled.toarray()
        
        y_labeled_bool = np.invert(np.isnan(y))

        delta = np.ones((n_docs,n_labels))
        for i in range(len(y)):
            if y[i] == 1:
                delta[i] = np.array([0,1])
            if y[i] == 0:
                delta[i] = np.array([1,0])
        
        likelihood = 1

        # print("fit starts")
        while iter < self.max_iter:
            
            likelihood_init = likelihood
            # E step
            probs = self.predict_proba(X_unlabeled)
            j = 0
            for i in range(len(y)):
                if np.isnan(y[i])==True:
                    delta[i] = probs[j,:]
                    j+=1
                    
            # print("delta from fit")
            # print(delta)
            self.delta = delta
            

            iter +=1

            # M step
            
            
            self.p_y = np.array([np.sum(delta[:,1])/delta.shape[0]])

            
            
            delta_1 = np.array([delta[:,1]]).T
            
            k = X.toarray()*delta_1
            K_1 = np.sum(k,axis=0) + self.smoothing
            M_1= np.sum(K_1) + self.smoothing*vocab_size

            
            K_0 = np.sum(X.toarray()*(np.array([delta[:,0]]).T),axis=0) + self.smoothing
            M_0= np.sum(K_0) + self.smoothing*vocab_size
            
            self.p_x_y = np.vstack((K_0/M_0,K_1/M_1)).T
            self.p_x_y = np.log(self.p_x_y)

            likelihood = self.likelihood(X,y)

            if abs(likelihood-likelihood_init)<0.1:
                break
            
        # print("fit ends")

        

    def likelihood(self, X, y):
        """
        Using fit self.p_y and self.p_x_y, compute the likelihood of the data.
            You should use logs to avoid underflow.
            This function *should* use unlabeled data.

        For unlabeled data, we define `delta(y | i) = p(y | x_i)` using the
            previously-learned p(x|y) and p(y) necessary to compute
            that probability. For labeled data, we define `delta(y | i)`
            as 1 if `y_i = y` and 0 otherwise; this is because for labeled data,
            the probability that the ith example has label y_i is 1.
            Following http://www.cs.columbia.edu/~mcollins/em.pdf,
            the log likelihood of the data can be written as:

            `sum_i sum_y (delta(y | i) * (log p(y) + sum_j log p(x_{i,j} | y)))`

        Note: If the word w appears `k` times in a document, the term
            `p(w | y)` should appear `k` times in the likelihood for that document!

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the (log) likelihood of the data.
        """

        assert hasattr(self, "p_y") and hasattr(self, "p_x_y"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2

        # unlabeled data 
        X_unlabeled = X[np.isnan(y)]
        # delta = self.delta
        delta_unlabeled = self.predict_proba(X_unlabeled)
        # print(delta_unlabeled)
        delta = np.zeros((X.shape[0],n_labels))
        
        j = 0
        for i in range(len(y)):
            if y[i] == 0:
                delta[i] = np.array([1,0])
            if y[i] == 1:
                delta[i] = np.array([0,1])
            if np.isnan(y[i]) == True:
                delta[i] = delta_unlabeled[j,:]
                j+=1
        
        
        sum_y_0 = np.sum(delta[:,0]*(np.log(1-self.p_y[0])+ X@self.p_x_y[:,0]))
        sum_y_1 = np.sum(delta[:,1]*(np.log(self.p_y[0])+ X@self.p_x_y[:,1]))
        
        
        sum_unlabeled = sum_y_0 + sum_y_1

        # print(sum_unlabeled)
        return sum_unlabeled
        
