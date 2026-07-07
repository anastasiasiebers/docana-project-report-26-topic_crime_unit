# Topic Lost or Found? 
## Measuring Topic Fidelity in Reddit TL;DR Summaries

*Group members: Ekaterina Kabashko (Student-ID: 1474411), Anastasia Siebers (Student-ID: 1498118)*

### Introduction

A struggle us humans face every day is: the days are too short, and in the digital age there is always more content than we have time to consume. On Reddit alone, in the first half of 2025, a total of about 23.7 million posts were created, which makes roughly 130,000 posts per day [[1]](#ref-1). These posts can be interesting and worth reading, but of course not all of these hundreds of thousands of posts per day are relevant to us.

This is where TL;DR summaries come in. They allow us to quickly decide whether a post is worth our time before reading the entire text.

However, this only works if the summary actually reflects the topics discussed in the original post. Imagine someone writes a long post about moving to a new city, feeling lonely, making new friends, and changing careers. If the TL;DR simply says "Moved to a new city and experienced a lot of changes," the topic of career change may be lost completely. As a result, someone interested in career advice might decide not to read the post, even though it contains information they would have found valuable.

So summing up the issue with TL;DRs might be that since TL;DRs are user-written, they may be short, selective, humorous, vague, or incomplete, omitting important topics. This can affect whether readers end up reading the full post.

In order to further investigate how big of an issue this is in reality, we are going to use the dataset *Webis-TLDR-17*, introduced by Völske et al. (2017) [[2]](#ref-2), which contains Reddit posts with author-written TL;DR summaries. It was introduced as a large-scale summarization dataset containing approximately 4 million content-summary pairs.

Previous work used this dataset for training and evaluation of abstractive summarization by machines [[2]](#ref-2). We opted for a slightly different perspective. Inspired by the work of Kim et al. (2024) [[3]](#ref-3), who study faithfulness and content selection in summaries, we came up with the idea that instead of generating these summaries with the help of machines, we would inspect whether the existing user-written summaries preserve the topic of the source post. This is what we will call topic fidelity from now on.

Since a manual comparison of the content and the summary would be tedious and would take way too long, we will take advantage of the NLP method of topic modeling, since topic models represent documents through latent themes/topics. This lets us compare the topic representation of a full post with a TL;DR.

We will use LDA as a classical probabilistic topic model, following Blei et al. (2003) [[4]](#ref-4), and BERTopic as a modern embedding-based topic modeling method introduced by Grootendorst (2022) [[5]](#ref-5).

The choice of the two methods is not random. One of our goals is also to conduct a method comparison to see how NLP methods evolved over time and whether the two approaches behave differently. We also decided on this approach knowing that short texts, like some of the TL;DRs that only contain 5-6 words, can be challenging for topic modeling. This makes the comparison even more meaningful, since we can investigate how the two models handle very short summaries. Recent work comparing topic modeling approaches has shown that BERTopic can produce coherent and interpretable topics for short or social-media-like texts, while classical probabilistic models remain useful and interpretable (Kaur & Wallace, 2024 [[6]](#ref-6); Jiang et al., 2026 [[7]](#ref-7)). We will explore whether we observe similar patterns in our project.

So our end goal is going to be to see whether the topics in the summary match the topics of the original post. At the end, we want to answer:

> **Main research question:**  
> Do Reddit TL;DR summaries preserve the topics of the original posts?

We want to go a bit more in depth and also figure out, as aforementioned, whether LDA and BERTopic produce different estimates of topic match. We also want to check whether short summaries are associated with lower topic match.

In order to answer these questions, we will sample Reddit post-summary pairs, train topic models on the full post texts in our sampled dataset, and then get topic probability distribution vectors for both the full post and the summary. We will then compare these vector representations of the topic probability distributions by calculating cosine similarity. We will compare the results of LDA with the results of BERTopic. To answer the last question, whether short summaries are associated with lower topic match, we will look at the correlation between summary length and the cosine similarity of the full post and summary.

### Dataset

We use the Webis-TLDR-17 dataset by Völske et al. (2017), which contains Reddit posts paired with author-written TL;DR summaries. In our project, `content` is used as the source document and `summary` as the corresponding TL;DR.

The dataset is suitable for our research question because each example directly links a full Reddit post with a human-written summary. This allows us to compare the topics in the original post with the topics preserved in the summary and evaluate topic fidelity at the content-summary pair level.

### Methods

#### Setup 

The experiments were implemented in Python using Google Colab and Jupyter notebooks.

Dataset access and preprocessing were performed in Google Colab. We used Colab because the Webis-TLDR-17 loading script downloads the original `corpus-webis-tldr-17.zip` archive, which is about 3.1 GB. Even when loading a smaller subset, access to the full archive may be required first. Colab provided sufficient temporary storage and avoided storing the archive locally.

The later topic modeling and analysis steps were conducted in a local Jupyter notebook environment. The analysis notebook was run with Python 3.13.9, while the preprocessing notebook was run in Google Colab with Python 3.12. No GPU was required for the experiments.

To make the project reproducible, all required Python libraries and their versions are listed in `requirements.txt`. The environment can be recreated with:

```bash
conda create --name docana-tldr python=3.13
conda activate docana-tldr
pip install -r requirements.txt
```

#### Experiments

##### Preprocessing

We first inspected a subset of the Webis-TLDR-17 training data to obtain subreddit-level counts. Based on this overview, we selected four subreddits with at least 1,000 valid content-summary pairs: `politics`, `gaming`, `todayilearned`, and `explainlikeimfive`.

From each subreddit, we randomly sampled 1,000 examples using a fixed random seed, resulting in a working dataset of 4,000 content-summary pairs. We kept the original dataset columns and added additional columns for cleaned and preprocessed text.

We created two preprocessing versions of the text. For BERTopic, we applied light cleaning to both `content` and `summary`: URL/link removal, HTML-like markup removal, Reddit Markdown link cleanup, and whitespace normalization. The resulting columns were `content_clean` and `summary_clean`. We did not apply stopword removal, lemmatization, or manual tokenization for this version, because BERT-based models use their own tokenizer and rely on natural sentence context.

For LDA, we applied stronger preprocessing to the cleaned text. This included lowercasing, tokenization, removal of non-alphabetic characters, stopword removal, removal of very short tokens, and lemmatization. The resulting columns were `content_lda_preprocessed` and `summary_lda_preprocessed`. This version was used for LDA because it reduces vocabulary sparsity and is better suited to a bag-of-words topic model.

##### Model training

To choose the final configuration, we tested three model setups:

| Run | LDA setting | BERTopic setting | Purpose |
|---|---|---|---|
| Run 1 | 100 topics | HDBSCAN min_cluster_size = 30 | Baseline |
| Run 2 | 50 topics | HDBSCAN min_cluster_size = 15 | More granular BERTopic, medium LDA |
| Run 3 | 25 topics | HDBSCAN min_cluster_size = 50 | Broader BERTopic |
| Final setup | 50 topics | HDBSCAN min_cluster_size = 50 | Final configuration |

For the LDA experiment, we trained a Gensim LDA model on the preprocessed full Reddit posts. The model used 50 topics, `random_state=100`, `passes=10`, `chunksize=10000`, and `alpha="auto"`. Words appearing fewer than 20 times in the corpus were removed from the dictionary. After training the model on the full posts, we inferred topic distributions for both posts and summaries in the same 50-topic space. This ensured that the topic vectors of a post and its TL;DR were directly comparable.

For the BERTopic experiment, we trained BERTopic on the lightly cleaned full Reddit posts and then transformed the summaries into the same topic space. We used the sentence-transformer model `all-MiniLM-L6-v2` for embeddings. The CountVectorizer used English stopwords, `min_df=2`, and an n-gram range of `(1, 2)`. UMAP was configured with `n_neighbors=15`, `n_components=5`, `min_dist=0.0`, `metric="cosine"`, and `random_state=265`. HDBSCAN was configured with `min_cluster_size=50`, `min_samples=5`, `metric="euclidean"`, `cluster_selection_method="eom"`, and `prediction_data=True`.

##### Evaluation

For each post-summary pair, we computed cosine similarity between the topic vector of the full post and the topic vector of the TL;DR summary. A higher cosine similarity indicates stronger topical alignment between the full post and its summary.

Finally, we compared the similarity distributions produced by LDA and BERTopic using histograms and boxplots. We also calculated Spearman correlation between summary length and cosine similarity to investigate whether shorter summaries were associated with weaker topic match.

### Results and Discussion

### Results and Discussion

The cosine similarity distributions show that LDA and BERTopic produce clearly different patterns of topic match between full Reddit posts and their TL;DR summaries.

<img width="1402" height="430" alt="image" src="https://github.com/user-attachments/assets/6a53728c-f0e9-4956-bd2f-6557b87e9760" />

For LDA, the histogram is strongly skewed toward very low cosine similarity. A large number of post-summary pairs receive scores close to zero. This suggests that, according to the LDA topic representation, many TL;DRs do not closely match the topic distribution of their original posts. One possible explanation is that LDA relies on bag-of-words representations and therefore struggles with very short summaries. If a summary contains only a few words, there may not be enough lexical evidence for LDA to recover the same topic distribution as in the full post.

BERTopic shows a different pattern. Its similarity distribution is more polarized: many summaries receive very high similarity scores, while another group receives very low scores, with fewer cases in the middle. This suggests that BERTopic often classifies summaries as either clearly aligned or clearly not aligned with the original post. Since BERTopic uses sentence embeddings, it can capture semantic similarity even when the summary uses different words from the original post. At the same time, very vague or extremely short summaries may still fail to preserve enough topical information and therefore receive low similarity scores.

This difference between LDA and BERTopic is an important finding. It shows that topic fidelity is not measured in exactly the same way by different topic modeling methods. LDA and BERTopic do not simply produce interchangeable results; the method chosen affects the interpretation of whether a TL;DR preserves the topics of the full post.

We also examined the relationship between summary length and topic similarity. The Spearman correlation between summary length and cosine similarity was weak but positive for both models. In the final setup, the correlation was approximately 0.26 for LDA and 0.16 for BERTopic. This means that longer summaries tend to have slightly higher topic similarity, but the relationship is not strong. In other words, longer summaries are not automatically faithful, and shorter summaries are not automatically unfaithful. However, longer summaries usually provide more space to mention multiple topics, making it easier for topic modeling methods to recover the topical overlap.

| Model | Spearman correlation |
|---|---:|
| BERTopic | approx. 0.2253 |
| LDA | approx. 0.1596 |

Overall, the results answer our main research question with a nuanced conclusion: Reddit TL;DRs do preserve topics in some cases, but topic preservation is inconsistent. Some summaries clearly overlap with their original posts, while others lose important topical information. The comparison between LDA and BERTopic further shows that this conclusion depends partly on the modeling approach used to represent topics.

### Conclusion

Summarize the major outcomes of your project, reflect on the research findings, and clearly state the conclusions you've drawn from the study.

### Contributions

| Team Member  | Contributions                                             |
|--------------|-----------------------------------------------------------|
| Ekaterina Kabashko  | |                                                       |
| Anastasia Siebers  | ...                                                       |

### References

<a id="ref-1"></a>
[1] Statista. (2026). *Number of Reddit posts created worldwide from 2018 to 2025*.  
https://www.statista.com/statistics/1319008/reddit-content-created/ (Last accessed: 22 June 2026).

<a id="ref-2"></a>
[2] Völske, M., Potthast, M., Syed, S., & Stein, B. (2017). *TL;DR: Mining Reddit to Learn Automatic Summarization*. In Proceedings of the Workshop on New Frontiers in Summarization, 59-63. Association for Computational Linguistics.  

<a id="ref-3"></a>
[3] Kim, Y., Chang, Y., Karpinska, M., Garimella, A., Manjunatha, V., Lo, K., Goyal, T., & Iyyer, M. (2024). *FABLES: Evaluating faithfulness and content selection in book-length summarization*. Proceedings of the 1st Conference on Language Modeling.  

<a id="ref-4"></a>
[4] Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation*. Journal of Machine Learning Research, 3, 993-1022.  

<a id="ref-5"></a>
[5] Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*.

<a id="ref-6"></a>
[6] Kaur, A., & Wallace, J. R. (2024). *Moving Beyond LDA: A Comparison of Unsupervised Topic Modelling Techniques for Qualitative Data Analysis of Online Communities*. 

<a id="ref-7"></a>
[7] Jiang, Y., Liu, S., & Fisher, P. A. (2026). *A Comparative Evaluation of Structural Topic Models and BERTopic for Short, Open-Ended Survey Responses*.   
