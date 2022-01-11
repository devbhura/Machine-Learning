def build_small_dataset():
    from src.data import build_dataset
    from src.data import train_test_unlabeled_split

    # Build the train/test split dictionary
    splits_dict = {
        "2009_Obama": 0, "2017_Trump": 0,
        "2016_Obama": 1, "2020_Trump": 1,
        "1993_Clinton": 1, "2000_Clinton": 1,
        "2001_Bush": 1, "2008_Bush": 1,
        "1989_Bush": 1, "1992_Bush": 1,
        "1981_Reagan": 1, "1988_Reagan": 1,
    }
    for year in range(2010, 2016):
        splits_dict[f"{year}_Obama"] = 2
    for year in range(2018, 2020):
        splits_dict[f"{year}_Trump"] = 2
    for year in range(1994, 2000):
        splits_dict[f"{year}_Clinton"] = 2
    for year in range(2002, 2008):
        splits_dict[f"{year}_Bush"] = 2
    for year in range(1990, 1992):
        splits_dict[f"{year}_Bush"] = 2
    for year in range(1982, 1988):
        splits_dict[f"{year}_Reagan"] = 2

    # Build and split the dataset
    data, labels, speeches, vocab = build_dataset(
          "data/", num_docs=40, max_words=50, vocab_size=10)
    return train_test_unlabeled_split(data, labels, speeches, splits_dict)
