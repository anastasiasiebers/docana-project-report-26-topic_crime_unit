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

We use the Webis-TLDR-17 dataset by Völske et al. (2017), which contains Reddit posts paired with author-written TL;DR summaries. In our project, content is used as the source document and summary as the corresponding TL;DR.

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

#### Experiments

We used Google Colab for dataset access and preprocessing because the Webis-TLDR-17 loading script downloads the original corpus-webis-tldr-17.zip archive, which is about 3.1 GB. Even when requesting a smaller subset, access to the full archive may be required first. Running this step in Colab avoids storing the archive locally and provides sufficient temporary storage for downloading and processing the data.

We first inspected a subset of the Webis-TLDR-17 training data to obtain subreddit-level counts. Based on this overview, we selected four subreddits with at least 1,000 valid content-summary pairs: politics, gaming, AdviceAnimals, and explainlikeimfive.
From each subreddit, we randomly sampled 1,000 examples, resulting in a working dataset of 4,000 content-summary pairs. We kept all original columns and added content_len and summary_len, which store the character lengths of the original post and TL;DR summary.
We created two preprocessing versions of the text. For BERTopic, we applied light cleaning to content and summary: URL/link removal, HTML-like markup removal, and whitespace normalization. The resulting columns are content_clean and summary_clean. No stopword removal, lemmatization, or manual tokenization was applied for this version, because BERT-based models use their own tokenizer and rely on natural sentence context.

For LDA, we applied stronger preprocessing to the cleaned text: lowercasing, tokenization, stopword removal, removal of very short tokens, and lemmatization. The resulting columns are content_lda_preprocessed and summary_lda_preprocessed. This version is used for LDA because it reduces vocabulary sparsity and is better suited to a bag-of-words topic model.


### Results and Discussion

Present the findings from your experiments, supported by visual or statistical evidence. Discuss how these results address your main research question.

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
