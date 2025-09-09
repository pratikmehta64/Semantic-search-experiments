Semantic-search application originally meant as an interview take-home

# Steps to prepare:
[Optional]
1. `python3 -m venv interview-env`
2. `source interview-env/bin/activate`

[Required]
3. `pip3 install -r requirements.txt`
4. `python3 migrate.py`

# Setup script: init.py
- Please set the three API keys in .env first, then run init.py:
`OPENAI_API_KEY`
`TURBOPUFFER_API_KEY`
`VOYAGEAI_API_KEY`
5. `python3 init.py`

# Retrieval script:
- Code that takes a query and returns up to 10 candidate IDs
- Contains an example of calling the eval API
6. Have query/queries stay in queries.json in the following format:
###
[
    "query": {
        "Title": <Your-Title>",
        "Natural Language Description": "<Your-Description>",
        "Hard Criteria": "<Your-hard-criteria>",
        "Soft Criteria": "<Your-soft-criteria>",
        "Yaml File": "<Yaml-Filename>"
    }
]
###
7. `python retrieval.py ./results/exp-4
    Usage: python retrieval.py <results-dir>

# Approach
<!-- Alert: There are no type checks in the code, so please exercise caution if editing it. Suffice to say, this code is NOT PROD-FRIENDLY -->
1. Baseline: 
- Default doc-embedding store provided via migrate.py
- Query: Title, Description, and Hard Criteria 
- Cosine similarity
- For this baseline as for other experiments, vector search results were compared to full-text-only search (FTS). 
- FTS outperformed vector search on hard-criteria and total avg scores nearly always (and was therefore used for the grading submission). This is likely attributable to the nature of the dataset and queries - similarity search was adequate given the likelihood of exact matches, though not so for the soft criteria.

2. Experiment 1:
- Query: Title, Description, Hard Criteria, and whatever portion of Soft Criteria fits in char limit (1024c)

3. Experiment 2:
- Query: No 'prompt' or 'Description'. Only Title, Hard Criteria, and Soft Criteria

4. Experiment 3:
- Query: No 'prompt'. Only Title, Hard Criteria, Soft Criteria and whatever portion of Description fits in char limit (1024c)
This seemed to work best among the alternatives.

5. Experiments 4-8:
Other variations that borrow the query either from experiment #3 or the baseline.

Three main submissions:
1. Submission #1: Vector_search
2. Submission #2: Full-text search
3. Submission #3: FTS + Vector_search + Reranking using reciprocal rank fusion

PLEASE REFER TO THE .PNGs in results/ For a comparision of scores across these three (subtle) variations for both full-text-search and vector-search.

Overall, the baseline version of the query with reranking using Reciprocal rank fusion over ANN+Full text search retrieval results provided best results, and constitute the final submission.
