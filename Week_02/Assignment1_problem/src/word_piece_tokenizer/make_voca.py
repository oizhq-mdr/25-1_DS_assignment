import math 
from collections import defaultdict, Counter
from pathlib import Path
from utils import whitespace_tokenize

def get_corpus(file_path):
    """파일에서 모든 텍스트 라인을 읽어 리스트로 반환합니다."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines

def get_initial_vocab(corpus):
    """
    초기 어휘 구성:
    각 단어를 문자 단위로 분해한 후 단어 경계를 나타내는 </w>를 붙여 vocabulary를 구성합니다.
    예: "hello" -> "h e l l o </w>"
    """
    vocab = Counter()
    for line in corpus:
        words = whitespace_tokenize(line)
        for word in words:
            tokens = list(word) + ["</w>"]
            token_seq = " ".join(tokens)
            vocab[token_seq] += 1
    return vocab

def get_pair_stats(vocab):
    """
    현재 vocabulary에서 각 단어(토큰 시퀀스) 내 인접한 토큰 쌍의 빈도를 계산합니다.
    """
    pairs = defaultdict(int)
    # TODO: 각 token_seq에서 인접한 토큰 쌍을 추출하여, 전체 vocabulary에서의 빈도를 계산하는 코드를 작성하세요.
    # 예: for token_seq, freq in vocab.items(): ... 
    for token_seq, freq in vocab.items():
        tokens = token_seq.split()
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += freq
    return pairs

def get_unigram_counts(vocab):
    """
    vocabulary 내의 모든 토큰(단일 문자 또는 병합된 토큰)의 빈도를 계산합니다.
    """
    unigram_counts = Counter()
    for token_seq, freq in vocab.items():
        tokens = token_seq.split()
        for token in tokens:
            unigram_counts[token] += freq
    return unigram_counts

def merge_vocab(pair, vocab):
    """
    주어진 pair(예: ("a", "b"))를 vocabulary의 모든 항목에서 병합합니다.
    "a b" 형태의 bigram을 "ab"로 치환합니다.
    """
    merged_vocab = {}
    pair_str = " ".join(pair)
    replacement = "".join(pair)
    # TODO: 주어진 pair를 활용하여 vocab 내의 모든 token sequence에서 "a b"를 "ab"로 병합하는 코드를 작성하세요.
    for token_seq, freq in vocab.items():
        tokens = token_seq.split()
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(replacement)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_token_seq = " ".join(new_tokens)
        merged_vocab[new_token_seq] = freq
    return merged_vocab

def compute_likelihood_score(pair, pair_freq, unigram_counts, total_tokens):
    """
    likelihood 점수 계산:
    - 관측 빈도: pair_freq
    - 기대 빈도: (freq(token1) * freq(token2)) / total_tokens
    점수는 observed * log(observed/expected)로 계산합니다.
    """
    # TODO: likelihood 점수를 계산하는 코드를 작성하세요.
    token1, token2 = pair
    observed = pair_freq
    freq1 = unigram_counts[token1]
    freq2 = unigram_counts[token2]

    expected = (freq1 * freq2) / total_tokens

    if observed == 0 or expected == 0:
        return 0
    
    score = observed * math.log(observed/expected)
    return score

def learn_wordpiece_vocab(file_path, num_merges=1000, target_vocab_size=1000):
    """
    test.txt 파일을 기반으로 likelihood 기반의 점수를 사용해 vocabulary를 학습합니다.
    
    num_merges: 최대 병합 횟수
    target_vocab_size: 원하는 최종 vocabulary 크기 (특수 토큰 포함)
    """
    corpus = get_corpus(file_path)
    vocab = get_initial_vocab(corpus)
    merges = []

    for i in range(num_merges):
        pair_stats = get_pair_stats(vocab)
        unigram_counts = get_unigram_counts(vocab)
        total_tokens = sum(unigram_counts.values())
        
        # 각 후보 쌍에 대해 likelihood 점수를 계산하는 부분
        scores = {}
        # TODO: pair_stats를 순회하면서 각 pair에 대한 likelihood 점수를 계산하고 scores 딕셔너리에 저장하는 코드를 작성하세요.
        for pair, freq in pair_stats.items():
            score = compute_likelihood_score(pair, freq, unigram_counts, total_tokens)
            scores[pair] = score
        if not scores:
            break
        
        # TODO: scores 딕셔너리에서 가장 높은 점수를 가진 pair(best_pair)와 그 점수(best_score)를 결정하는 코드를 작성하세요.
        best_pair = None   # 예시: best_pair = max(scores, key=scores.get)
        best_score = 0     # 예시: best_score = scores[best_pair]
        best_pair = max(scores, key=scores.get)
        best_score = scores[best_pair]
        
        # 점수가 음수이거나 변화가 없으면 종료
        if best_score <= 0:
            break
        
        merges.append(best_pair)
        vocab = merge_vocab(best_pair, vocab)
        
        # 현재 vocabulary 크기 확인 (특수 토큰 제외)
        current_vocab = set()
        for token_seq in vocab.keys():
            tokens = token_seq.split()
            for token in tokens:
                token = token.replace("</w>", "")
                current_vocab.add(token)
        if len(current_vocab) >= target_vocab_size:
            break

    return current_vocab, merges

def save_vocab(vocab_set, output_path="vocab.txt"):
    """
    생성된 vocabulary를 파일로 저장합니다.
    특수 토큰([UNK], [CLS], [SEP])을 우선 추가합니다.
    """
    special_tokens = ["[UNK]", "[CLS]", "[SEP]"]
    with open(output_path, "w", encoding="utf-8") as f:
        for token in special_tokens:
            f.write(token + "\n")
        for token in sorted(vocab_set):
            f.write(token + "\n")

if __name__ == "__main__":
    # test.txt 파일을 기반으로 vocabulary 생성
    file_path = Path(__file__).resolve().parent.parent.parent / "tests" / "tests.txt"
    vocab_set, merges = learn_wordpiece_vocab(str(file_path), num_merges=1000, target_vocab_size=1000)
    save_vocab(vocab_set, output_path="src/word_piece_tokenizer/vocab.txt")
    print("Vocabulary 생성 완료. 총 토큰 수:", len(vocab_set))
    print("병합 기록 (일부):", merges[:10])
