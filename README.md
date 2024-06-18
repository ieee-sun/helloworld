### hello world!
Testing the first repo

---

### Q - should I put `.py` or only Jupyter notebooks (`.ipynb`) in Github?
##### by [Melonface](https://www.reddit.com/user/MelonFace/): 
> how I'd say you do it. I work in the industy.
> 1. Develop in jupyter / google colab.
>
> 2. Once it runs, find things or patterns you're doing many times, make functions for those in a `utils.py` or similar collection of tools you may reuse in the future.
>
> 3. Take any non-general functions in the notebook and move them to a `my_project_stuff.py` (obviously better name).
>     * `%load_ext autoreload native jupyter` plugin will let you change these functions without restarting the kernel.
>
> 5. Parameterize and move the model definition into `name_model.py`
> 6. Now your notebook should contain only the data loading and training code. This is fine. And still gives you the debug/experiment freedom.
>
> 7. Personally I also put the training code in a `.py` file at the end and load checkpoints into notebooks as needed to debug or fine tune. But that's personal taste. The reason for this is the service I use to run training in cloud. Where it's more reliable to run long running python programs rather than jupyter kernels.

##### by [ktpr](https://www.reddit.com/user/ktpr/):
> You could convert it to a [CLI integration](https://docs.github.com/en/github-cli/github-cli/quickstart) , a cooperative layout on Github -
> ```
>    pip install -U pipenv cookiecutter
>    cookiecutter gh:crmne/cookiecutter-modern-datascience
> ```
> Copy your `.ipynb` notebooks into `./notebook/`
> - Step 1: from your notebooks, place data, pipeline and model code into their respective directories. Wrap with Typer for CLI integration
> - Step 2: in `./serve`, make an `app.py` file that calls appropriate code to expose your ML project as an API. 
> - Endpoints can load, train, predict and otherwise serve your project.

---
### Q -  train-test splitting and evaluate performance metrics when creating a recommender system using collaborative filtering?

Answer by MIT expert:

> There are two main approaches to consider:
> 
> **Holdout**: This is the simplest method. Split your data into training and testing sets (common split is 80/20). Train your model on the training set and evaluate its performance on the unseen testing set.
> 
> **Cross-Validation**: This is a more robust approach. Divide your data into several folds (e.g., 10 folds). Train your model on all folds except one (validation fold), and evaluate on the validation fold. Repeat this process for each fold, averaging the results for a more reliable estimate.
> 
> #### Important Considerations:
> 
> **User-based** vs. **Item-based Splitting**: Decide whether to split by users or items. User-based splitting ensures users have interactions in both sets, but might not expose the model to unseen items. Item-based splitting can handle unseen items, but users might not have interactions in both sets.
> 
> **Cold Start Problem**: New users or items might not have enough data in the training set for accurate recommendations. Consider alternative approaches like item popularity or content-based filtering for these cases.
> 
> Here are some common metrics used to evaluate recommender systems:
> 
> **Root Mean Squared Error** (RMSE): Measures the average magnitude of the difference between predicted and actual ratings.
> 
> **Mean Absolute Error** (MAE): Similar to RMSE but uses absolute differences.
> 
> `Precision@k` and `Recall@k`: Measure how many relevant items are recommended among the top k recommendations.
> 
> **Normalized Discounted Cumulative Gain** (NDCG): Measures the ranking quality of recommendations, considering items with higher relevance being positioned higher.
> 
> #### Choosing the Right Metric:
> 
> The best metric depends on your specific goals. RMSE and MAE are good for predicting ratings, while `Precision@k` and `Recall@k` are better for capturing relevance. NDCG prioritizes highly relevant items.


### INFO - Python and Data Sciences YouTube Channels

1. [Data Analysis Using SPSS (49 videos)](https://lnkd.in/dE8e7H8R)
2. [Data Analysis Using Jamovi (26 videos)](https://lnkd.in/d4d_w9NR)
3. [Data Analysis Using Smart PLS-4 (35 videos)](https://lnkd.in/d-ZJ-Vkr)
4. [Time Series Analysis Using Eviews-12 (32 videos)](https://lnkd.in/dNthyy55)
5. [Panel Data Analysis Using Eviews-12 (25 videos)](https://lnkd.in/d2-KKKxV)
6. [Data Analysis Using Python (37 videos)](https://lnkd.in/dX8HDdxC)
7. [Data Analysis Using Rapidminer (70 videos)](https://lnkd.in/dpNTfA_X)
8. [Data Science Using R and R-Studio (28 videos)](https://lnkd.in/ddqQVHTg)
9. [Structural Equation Modeling using Smart PLS (22 videos)](https://lnkd.in/df2dzu7d)
10. [Qualitative Data Analysis in Nvivo (17 videos)](https://lnkd.in/deXZacwp)
11. [Machine Learning Using Python (15 videos)](https://lnkd.in/dDRMzzCT)
12. [Structural Equation Modelling using ADANCO (15 videos)](https://lnkd.in/d4EDmhdk)
13. [Data Visualization using Tableau (100 videos)](https://lnkd.in/dctQQjey)
14. [Data Analysis Using KNIME (139 videos)](https://lnkd.in/dp-GnY2C)
15. [Structural Equation Modeling using IBM SPSS AMOS](https://lnkd.in/d2dSKc7i)
16. [Artificial Intelligence (AI) Tools in Research](https://lnkd.in/dV5r56rW)
17. [Data Analysis Using Alteryx](https://lnkd.in/dHBaF-Zf)
18. [Data Analysis Using Excel](https://lnkd.in/dj9BbNjB)
19. [Data Analysis Using Power BI](https://lnkd.in/dPRQicV5)

### Text operations Tutorial
###### by U of Penn. Library:
> https://guides.library.upenn.edu/penntdm/API

---
> ###### Prepared by:
> ###### Sun CHUNG, *SMIEEE* M.Sc. HKU

---
###### End of File.


