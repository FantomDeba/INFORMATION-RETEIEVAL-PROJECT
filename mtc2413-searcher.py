import sys
from pyserini.search.lucene import LuceneSearcher
from collections import defaultdict
import math
try:
    from pyserini.index.lucene import IndexReader 
except ImportError:
    try:
        from pyserini.index import IndexReader 
    except ImportError:
        IndexReader = None 
c = 1.0
alpha = 0.5
tau_x = 6
tau_y = 65
lambda_exp = 0.5

def parse_fire_queries(file_path):
    queries = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        qid, title = None, None
        for line in f:
            line = line.strip()
            if line.startswith("<num>"):
                qid = line.replace("<num>", "").replace("</num>", "").strip()
            elif line.startswith("<title>"):
                title = line[len("<title>"):].strip()
            elif line.startswith("</top>") and qid and title:
                queries[qid] = title
                qid, title = None, None  # Reset for next query
    return queries


def compute_normalized_tf(tf, doc_len, mean_tf):
    X = math.log(1 + tf) / math.log(c + mean_tf) if mean_tf > 0 else 0
    Y = tf * math.log(1 + 1000 / doc_len) if doc_len > 0 else 0
    return X, Y

def truncated_exp_score(x, tau, lambda_):
    if 0 < x < tau / 2:
        return (1 - math.exp(-lambda_ * x)) / (1 - math.exp(-lambda_ * tau))
    elif x <= tau:
        return x / tau
    else:
        return 0.0

class FallbackIDFCalculator:
    def __init__(self, searcher):
        self.searcher = searcher
        self.doc_count = self._estimate_doc_count()
        self.term_cache = {}

    def _estimate_doc_count(self):
        try:
            hits = self.searcher.search("the", k=1)
            return int(hits[0].score) if hits else 1000000
        except:
            return 1000000  

    def get_idf(self, term):
        if term in self.term_cache:
            return self.term_cache[term]
        try:
            hits = self.searcher.search(term, k=100)
            df = len(hits)
            idf = math.log((self.doc_count + 1) / (df + 1))
            self.term_cache[term] = idf
            return idf
        except:
            return 0  # Fallback if all else fails

def rerank(query, hits, searcher, idf_calculator):
    doc_scores = []
    
    for hit in hits:
        try:
            raw = searcher.doc(hit.docid).raw()
            content = raw if raw is not None else ""
            doc_words = content.split()
            doc_len = len(doc_words)

            tf_map = defaultdict(int)
            for word in doc_words:
                tf_map[word] += 1

            query_terms = [term for term in query.split() if term in tf_map]
            mean_tf = sum(tf_map[term] for term in query_terms) / len(query.split()) if query_terms else 0

            score = 0.0
            for term in query.split():
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                
                X, Y = compute_normalized_tf(tf, doc_len, mean_tf)
                f_xt = truncated_exp_score(X, tau_x, lambda_exp)
                f_yt = truncated_exp_score(Y, tau_y, lambda_exp)
                
                idf = idf_calculator.get_idf(term)
                score += idf * (alpha * f_xt + (1 - alpha) * f_yt)

            doc_scores.append((hit.docid, score))
        except Exception as e:
            print(f"Error processing document {hit.docid}: {str(e)}", file=sys.stderr)
            continue

    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 mtc2413-searcher.py <index_dir> <fire_query_file>")
        sys.exit(1)

    index_dir = sys.argv[1]
    query_file = sys.argv[2]
    
    try:
        searcher = LuceneSearcher(index_dir)
        searcher.set_bm25()
        if IndexReader is not None:
            try:
                index_reader = IndexReader(index_dir)
                idf_calculator = index_reader
            except Exception as e:
                print(f"Warning: Couldn't initialize IndexReader, using fallback method: {str(e)}", file=sys.stderr)
                idf_calculator = FallbackIDFCalculator(searcher)
        else:
            idf_calculator = FallbackIDFCalculator(searcher)
            
        queries = parse_fire_queries(query_file)

        for qid, query_text in queries.items():
            try:
                bm25_hits = searcher.search(query_text, k=2000)
                reranked = rerank(query_text, bm25_hits, searcher, idf_calculator)
                for rank, (docid, score) in enumerate(reranked[:1000], start=1):
                    print(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.4f}\tmtc2413")
            except Exception as e:
                print(f"Error processing query {qid}: {str(e)}", file=sys.stderr)
                continue
                
    except Exception as e:
        print(f"Fatal error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()