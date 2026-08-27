# English to French Machine Translation

## Selected Word-Embedding Method: Word2Vec
To upgrade the model's input representation, I replaced the frequency-based embeddings with **Word2Vec**. 
Unlike frequency counts (which treat words as isolated, orthogonal vectors), Word2Vec learns dense vector representations where words with similar meanings are mapped to nearby points in the vector space. 
* **Implementation:** I utilized Gensim to train separate Word2Vec models on the English and French training vocabularies (vector size = 256). 
* **Why Word2Vec?** It captures deep semantic and syntactic relationships (e.g., context and synonyms) which is critical for a translation task. The pre-trained Gensim weights were then injected directly into the PyTorch `nn.Embedding` layers of both the encoder and decoder.

## Architecture: BiLSTM & Attention Mechanism
The core architecture follows an Encoder-Decoder structure designed to handle variable-length sequences.

* **The Encoder (BiLSTM):** A standard LSTM reads data in one direction, but language context often relies on words that appear later in a sentence. I used a Bidirectional LSTM to read the English sequence from left-to-right *and* right-to-left. The hidden states from both directions are concatenated, providing a much richer, context-aware memory of the source sentence.
* **The Attention Mechanism:** In older Seq2Seq models, the entire input sequence is compressed into a single context vector, creating an information bottleneck. By implementing attention, the decoder can "look back" and dynamically calculate alignment scores for every word in the encoder's memory at each decoding step. A padding mask is applied so the model doesn't waste attention on `<pad>` tokens.
* **The Decoder (LSTM):** Initialized with the encoder's final state, the decoder generates the French sentence one token at a time, utilizing the attention context vector to decide which English words are most relevant for the current translation step.

## Preprocessing and Training Process
**Data Preprocessing:**
1. **Cleaning:** All text was lowercased, special characters were removed, and common contractions (e.g., "don't" -> "do not", "j'ai" -> "je ai") were expanded to reduce vocabulary sparsity.
2. **Vocabulary Building:** We built dictionaries mapping words to integer IDs, explicitly reserving slots for special tokens: `<pad>` (padding), `<unk>` (unknown), `<start>` (start of sequence), and `<end>` (end of sequence).
3. **Tensor Formatting:** Sentences were padded to a maximum length of 20 tokens to create uniform batch tensors.


## Evaluation Metrics: BLEU & ROUGE
To properly evaluate translation quality, we moved beyond simple token-accuracy and implemented standard MT metrics:
* **BLEU (Bilingual Evaluation Understudy):** Evaluates precision. It checks how many n-grams (1-gram to 4-grams) in our model's generated translation match the reference French translation. BLEU is excellent for measuring exact word matches and local fluency.
* **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Primarily evaluates recall. We used ROUGE-1 (unigrams), ROUGE-2 (bigrams), and ROUGE-L (Longest Common Subsequence) to ensure the generated translation captures the core structural meaning of the reference, even if the exact phrasing differs slightly.

## Results & Discussion

* **BLEU-1:** `55.04%`
* **BLEU-4:** `28.75%`
* **ROUGE-L:** `55.28%`

Attention Heatmap:
![Attention Heatmap](attentionHeatmap.png)

**Discussion:** 
The transition to Word2Vec embeddings allowed the model to handle synonyms and context much better than the baseline. While the BLEU-4 score indicates room for improvement on longer phrasal fluency, the BLEU-1 and ROUGE-L scores demonstrate that the model successfully captures the core meaning of the English inputs. The attention heatmaps further validate this, showing strong diagonal alignments between the source and target words.

The attention heatmap for "did he touch you" reveals a classic case of decoder looping, where the model starts strong but quickly loses its way. While it correctly aligns the initial English words "did" and "he" with the French "est ce qu il", its attention becomes entirely fixed on the word "touch." Ultimately, the model hallucinates the word "amusé" to break the cycle, completely failing to translate the sentense.