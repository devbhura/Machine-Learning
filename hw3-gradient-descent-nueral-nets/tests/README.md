# Rubric structure

`rubric.json` is structured as follows:

```
{
    "test_imports": {
            "weight": "required",
            "depends": []
        },
    "test_netid": {
        "weight": "required",
        "depends": []
    }
}
```

Each test case gets a weight. If it's required, it means they get a zero if they don't
pass this test case. Otherwise it gets added to their sum. You need more than one 
regular test case for the autograder to work. Depends is an array that checks if other
test cases have passed. If they haven't passed, that test case won't run.