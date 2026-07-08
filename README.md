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

Dataset access and preprocessing were performed in Google Colab. We used Colab because the Webis-TLDR-17 loading script downloads the original `corpus-webis-tldr-17.zip` archive, which is about 3.1 GB. Even when loading a smaller subset, access to the full archive is required first. Colab provided sufficient temporary storage and avoided storing the archive locally.

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

One challenge with the LDA preprocessing was that some TL;DR summaries were extremely short. After removing stopwords, punctuation, and very short tokens, 25 summaries became empty. We did not drop these cases, because the fact that no meaningful tokens remained is itself relevant for topic fidelity. Instead, we kept them in the dataset and assigned them a cosine similarity of 0 for the LDA-based comparison, since an empty topic representation cannot be considered topically similar to the original post.

The final preprocessed dataset was saved as `selected_4_subreddits_1000_each_clean_lda.csv` and included in the `code` folder of the repository.

##### Model Training

We used two topic modeling methods: LDA and BERTopic. LDA is a classical probabilistic topic model that works with word co-occurrence and bag-of-words representations. BERTopic is a more modern embedding-based method. It first represents texts through semantic embeddings and then clusters similar texts into topics. By comparing both methods, we could test whether topic fidelity depends on the type of topic representation.

One challenge was finding parameter settings that made the comparison between LDA and BERTopic fair. If one model is poorly configured, low similarity scores may reflect the setup rather than a real difference between the methods. We therefore tested several configurations before choosing the final setup.

| Run | LDA setting | BERTopic setting | Purpose |
|---|---|---|---|
| Run 1 | 100 topics | HDBSCAN `min_cluster_size=30` | Baseline |
| Run 2 | 50 topics | HDBSCAN `min_cluster_size=15` | Medium LDA, more granular BERTopic |
| Run 3 | 25 topics | HDBSCAN `min_cluster_size=50` | Broader topic setup |
| Final setup | 50 topics | HDBSCAN `min_cluster_size=50` | Final configuration |

For LDA, we varied the number of topics. This parameter controls how detailed the topic space is. If there are too many topics, related content may be split into very narrow topics. This can make it harder for short summaries to match the full post. If there are too few topics, different themes may be merged into topics that are too broad. We selected 50 topics for the final LDA setup because it provided a reasonable middle ground.

For BERTopic, we varied the HDBSCAN `min_cluster_size` parameter. This parameter controls how large a group of documents must be before it can become a topic. Smaller values create more and smaller clusters, while larger values create fewer and broader topics. We selected `min_cluster_size=50` because broader topics were more suitable for comparing long posts with short TL;DR summaries.

For the final LDA experiment, we trained a Gensim LDA model on the preprocessed full Reddit posts. The model used 50 topics, `random_state=100`, `passes=10`, `chunksize=10000`, and `alpha="auto"`. Words appearing fewer than 20 times in the corpus were removed from the dictionary to reduce noise from rare terms. After training the model on the full posts, we inferred topic distributions for both posts and summaries in the same 50-topic space.

For the final BERTopic experiment, we trained BERTopic on the lightly cleaned full Reddit posts. The summaries were then transformed into the same topic space. We used the sentence-transformer model `all-MiniLM-L6-v2` for embeddings. The CountVectorizer used English stopwords, `min_df=2`, and an n-gram range of `(1, 2)`, so the model could represent both single words and short phrases. UMAP was configured with `n_neighbors=15`, `n_components=5`, `min_dist=0.0`, `metric="cosine"`, and `random_state=265`. HDBSCAN was configured with `min_cluster_size=50`, `min_samples=5`, `metric="euclidean"`, `cluster_selection_method="eom"`, and `prediction_data=True`.

To make the comparison fair, we trained both models only on the full Reddit posts. The TL;DR summaries were then mapped into the same topic space. This way, each post and its summary could be compared using the same set of topics.

##### Evaluation

For each post-summary pair, we computed cosine similarity between the topic vector of the full post and the topic vector of the TL;DR summary. A higher cosine similarity indicates stronger topical alignment between the full post and its summary.

Finally, we compared the similarity distributions produced by LDA and BERTopic using histograms and boxplots. We also calculated Spearman correlation between summary length and cosine similarity to investigate whether shorter summaries were associated with weaker topic match.

### Results and Discussion

The cosine similarity histograms show that LDA and BERTopic behave quite differently when comparing full Reddit posts with their TL;DR summaries.

<p style="margin: 24px 0 32px 0;">
  <img width="1402" height="430" alt="Histogram of LDA and BERTopic cosine similarity" src="https://github.com/user-attachments/assets/6a53728c-f0e9-4956-bd2f-6557b87e9760" />
</p>

For LDA, many scores are close to zero. This means that, according to the LDA topic representation, many summaries are not very close to the topic distribution of their original posts. One reason for this may be that LDA works with word-based, bag-of-words representations. Very short summaries often contain too few words for LDA to recover the same topics as in the full post.

BERTopic shows a more polarized pattern. Many summaries receive very high similarity scores, while another group receives very low scores, with fewer cases in the middle. This makes sense because BERTopic uses embeddings and can capture semantic similarity even when the exact words differ. However, if a TL;DR is very vague or too short, it may still not contain enough topical information.

<p style="margin: 24px 0 32px 0;">
  <img width="1280" height="383" alt="Boxplots of LDA and BERTopic cosine similarity" src="https://github.com/user-attachments/assets/e984abab-812a-4257-b769-56936c7e5158" />
</p>

The boxplots show the same tendency as the histograms. LDA has a very low median similarity score, meaning that many summaries are only weakly aligned with their full posts. BERTopic has a much higher median and a wider range of scores. This supports our main finding that the two models produce different views of topic fidelity.

We also looked at whether longer summaries are more topically similar to their original posts. For this, we used Spearman correlation, because it is less sensitive to outliers and does not assume a perfectly linear relationship.

| Model | Spearman correlation |
|---|---:|
| BERTopic | approx. 0.2253 |
| LDA | approx. 0.1596 |

Both correlations are weak but positive. This means that longer summaries tend to have slightly higher topic similarity, but the effect is small. A longer TL;DR is not automatically more faithful, and a short TL;DR is not automatically worse. Still, longer summaries usually give authors more space to mention several topics, which makes topical overlap easier for the models to recover.

To make the similarity scores easier to interpret, we also inspected individual post-summary pairs. These examples show that high scores usually correspond to clear topical overlap, while low scores often occur when the TL;DR is very vague or only loosely connected to the full post.

| Case | Model | Similarity | Example summary | Interpretation |
|---|---|---:|---|---|
| High similarity | LDA | 0.9673 | “The producer is the movie's boss... The director is more of an artist...” | The summary preserves the central distinction discussed in the post. |
| High similarity | BERTopic | 1.0000 | “Punk Buster ruined BF2 for me...” | The summary clearly captures the post’s main topic: frustration with PunkBuster and BF2. |
| Low similarity | LDA | 0.0057 | “The ABC conjecture has no effect on your daily life.” | The summary keeps only a narrow conclusion from a more detailed mathematical explanation. |
| Low similarity | BERTopic | 0.0000 | “though? Fantasy is powerful :)” | The summary is too short and vague to preserve the post’s topic. |

### Conclusion

First, TL;DR fidelity is real, but inconsistent—and it depends heavily on how you measure it. Since LDA and BERTopic showed different patterns, we can't simply conclude that users always stay faithful to the original topic when summarizing, nor that they never do. When we looked at examples both models classified as highly similar, we did see a clear topical overlap. 

Second, absolute shortness hurts fidelity. Longer TL;DRs tend to better recover the topics of the full post. This makes intuitive sense, but it also suggests that even though a topic can in principle be captured in just a few words, people writing summaries often need more words to actually convey it.

Lastly, some open problems remain. We measured topic *overlap*, not semantic *faithfulness*—two texts can land in the same topic cluster while saying very different things within it. A natural next step would be to measure semantic similarity using sentence embeddings, rather than topic-distribution similarity. We focused on topic distributions here because we were specifically interested in whether the choice of NLP method—a more traditional probabilistic approach versus a more modern embedding-based one—leads to different conclusions. And the answer is yes—even measuring the exact same phenomenon, the method you choose shapes the story you tell - at least in our case.

### Contributions

| Team Member | Contributions |
|---|---|
| Ekaterina Kabashko | Data collection and preprocessing pipeline, parameter tuning |
| Anastasia Siebers | Analysis pipeline, model implementation and evaluation |
| Together | Results interpretation, final report and presentation |

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
