import pysolr
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from typing import List, Any, Tuple
from response_model import SolrResponse, Comment, Keyword

BASE_URL = "http://localhost:8983/solr/mycollection/"
model = SentenceTransformer('all-MiniLM-L6-v2')
nlp_en = spacy.load("en_core_web_sm") #spacy model for NLP tasks

def connect_to_solr():
    """
    Connect to solr
    """
    default_params = {
        "rows": 5,
        "sort": "score desc",
    }

    return pysolr.Solr(BASE_URL, always_commit=True, timeout=10), default_params

def get_keywords_from_spacy_and_tfidf(documents: List[str], top_k: int = 5) -> List[Keyword]:
    """
    Uses spacy for smarter tokenization and extract keywords from documentsu using tfidf as well
    :param documents: list of documents to extract keywords from
    :param top_k: number of top keywords to extract
    :return: list of keywords
    """
    processed_docs = []

    for doc in documents:
        spacy_doc = nlp_en(doc.lower())
        tokens = [token.lemma_ for token in spacy_doc if not token.is_stop and not token.is_punct and token.is_alpha]
        processed_docs.append(" ".join(tokens))

    # Use TF-IDF to extract keywords
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(processed_docs)
    feature_names = tfidf.get_feature_names_out()
    summed_tfidf = tfidf_matrix.sum(axis=0).A1

    scores = list(zip(feature_names, summed_tfidf))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    keywords = [Keyword(keyword=keyword, count=score) for keyword, score in sorted_scores]
    return keywords

def extract_keywords_semantic(query: str, documents: List[str], top_k: int = 5) -> List[Keyword]:
    """
    Uses sentence transformers to extract keywords from documents using semantic similarity
    :param query: query to extract keywords from
    :param documents: list of documents to extract keywords from
    :param top_k: number of top keywords to extract
    :return: list of keywords
    """
    # combine documents for simplicity
    full_text = " ".join(documents)
    spacy_doc = nlp_en(full_text.lower())
    candidates = list(set([
        token.lemma_ for token in spacy_doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]))

    # generate embeddings
    query_embedding = model.encode(query, convert_to_tensor=True)
    candidate_embeddings = model.encode(candidates, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, candidate_embeddings)[0]
    
    keyword_scores = list(zip(candidates, similarities.cpu().numpy()))
    sorted_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)[:top_k]

    keyword_counts = Counter([
        token.lemma_ for token in spacy_doc
        if token.lemma_ in dict(sorted_keywords)
    ])

    return [Keyword(keyword=keyword, count=keyword_counts[keyword]) for keyword, _ in sorted_keywords if keyword in keyword_counts]

def get_results(query: str, additional_params: dict = None) -> Tuple[SolrResponse, List[Keyword]]:
    """
    Get results and top keywords from solr
    :param query: query to search for
    :param additional_params: additional params to pass to solr
    :return: results from solr
    """
    solr, default_params = connect_to_solr()
    params = default_params.copy()

    # merge in additional query params if needed
    if additional_params:
        params.update(additional_params)

    query = "comment_content:" + query + " OR post_content:" + query
    try:
        results = solr.search(query, **params)
        raw_response = results.raw_response
        solr_response = raw_response["response"]

        documents = [doc.get("comment_content", "") for doc in solr_response["docs"]]
        
        # Step 1: Get TF-IDF-based keyword extraction
        tfidf_keywords = get_keywords_from_spacy_and_tfidf(documents)
        # Step 2: Get semantic keyword extraction
        semantic_keywords = extract_keywords_semantic(query, documents)
        # Step 3: Combine TF-IDF and semantic keyword extraction
        combined = Counter()
        for keyword_dict in tfidf_keywords + semantic_keywords:
            for word, count in keyword_dict.items():
                combined[word] += count

        # Step 4: Get top K keywords
        keywords = [Keyword(keyword=keyword, count=count) for keyword, count in combined.most_common(5)]

        return SolrResponse(**solr_response), keywords
    
    except pysolr.SolrError as e:
        print(f"Solr error: {e}")
        return SolrResponse(numFound=0, start=0, docs=[], keywords=[])
    
def convert_to_query_response(solr_response: SolrResponse) -> Tuple[int, List[str], List[Comment]]:
    """
    Convert solr response to query response
    """
    recordCount = solr_response["numFound"]
    #get unique subreddits from solr response
    subreddits = set()
    comments = []

    for doc in solr_response["docs"]:
        subreddits.add(doc["subreddit"])
        comments.append({
            "text": doc["comment_content"],
            "sentiment": "postitive" #TODO: add sentiment analysis
        })

    return recordCount, list(subreddits), comments