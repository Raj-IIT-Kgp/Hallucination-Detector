import nltk

def extract_claims(answer):
    """
    Extracts individual claims (sentences) from a paragraph using traditional NLP (NLTK).
    This ensures 100% deterministic extraction and makes it mathematically impossible 
    to hallucinate new claims during the extraction phase.
    """
    # Use NLTK's sentence tokenizer to split the text by punctuation.
    claims = nltk.sent_tokenize(answer)
    
    # Clean up whitespace just in case
    claims = [claim.strip() for claim in claims if claim.strip()]
    
    return claims