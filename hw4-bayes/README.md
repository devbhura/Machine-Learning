# HW 4: Naive Bayes and Expectation Maximization CS 349 @ NU

**IMPORTANT: PUT YOUR NETID IN THE FILE** `netid` in the root directory of the assignment. 
This is used to put the autograder output into Canvas. Please don't put someone else's netid 
here, we will check.

In this assignment, you will:
- Implement Naive Bayes for fully-labeled data
- Implement Naive Bayes for partially-labeled data using the EM algorithm
- Use your trained model to analyze a dataset of political speeches

## Clone this repository

To clone this repository install GIT on your computer and copy the link of the repository (find above at "Clone or Download") and enter in the command line:

``git clone YOUR-LINK``

Alternatively, just look at the link in your address bar if you're viewing this README in your submission repository in a browser. Once cloned, `cd` into the cloned repository. Every assignment has some files that you edit to complete it. 

## Files you edit

See problems.md for what files you will edit.

Do not edit anything in the `tests` directory. Files can be added to `tests` but files that exist already cannot be edited. Modifications to tests will be checked for.

## Environment setup

Make a conda environment for this assignment, and install the requirements. This should look something like:
- ``conda create -n hw4 python=3.9``
- ``conda activate hw4``
- ``pip install -r requirements.txt``

If you run into problems, try `conda upgrade conda` and repeat the above. If that doesn't work, open an issue or post on CampusWire.
Do not install or use additional packages with `conda` or `pip`; if your code relies on them, you may get a 0 on the assignment.

## Running the test cases

The test cases can be run from the root directory of the repository with:

``python -m pytest -s``

Note: *this is how your code will be graded!* If this command does not run (in
an environment with only the provided package requirements), you may get a 0 on the assignment.

## Questions? Problems? Issues?

Simply open an issue on the starter code repository for this
assignment [here](https://github.com/NUCS-349-Fall21/hw4-naive-bayes/issues).
Someone from the teaching staff will get back to you through there!
